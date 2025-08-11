from typing import Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import matplotlib.cm as cm

# =========================
# Effect specifications
# =========================

EffectKind = Literal["log1p", "sqrt", "linear", "sigmoid", "power", "exp_sat", "hill"]


class EffectSpec(BaseModel):
    """Parametric effect of spend on conversions (before noise)."""

    kind: EffectKind
    scale: float = 1.0  # multiplicative factor
    # optional params per kind
    # linear: y = scale * (a * x + b)
    a: float = 1.0
    b: float = 0.0
    # power:  y = scale * (x ** p)
    p: float = 0.5
    # sigmoid: y = scale * (1 / (1 + exp(-k*(x - x0))))
    k: float = 0.1
    x0: float = 0.0

    def apply(self, x: np.ndarray) -> np.ndarray:
        if self.kind == "log1p":
            return self.scale * np.log1p(x)
        elif self.kind == "sqrt":
            return self.scale * np.sqrt(x)
        elif self.kind == "linear":
            return self.a * x + self.b
        elif self.kind == "sigmoid":
            return self.scale * (1.0 / (1.0 + np.exp(-self.k * (x - self.x0))))
        elif self.kind == "power":
            return self.scale * (x**self.p)
        elif self.kind == "exp_sat":
            return self.scale * (1.0 - np.exp(-self.k * x))
        elif self.kind == "hill":
            return self.scale * (x**self.p) / (self.x0**self.p + x**self.p)
        else:
            raise ValueError(f"Unknown effect kind: {self.kind}")


# =========================
# Channel / interaction config
# =========================


class ChannelConfig(BaseModel):
    name: str
    spend_range: tuple[float, float] = Field(..., description="Uniform low/high")
    effect: EffectSpec
    noise_scale: float = 0.0
    lag_days: int = 0  # if >0, effect applied after lag


class InteractionSpec(BaseModel):
    """Simple pairwise interaction; extend as needed."""

    name: str
    var1: str
    var2: str
    kind: Literal["product"] = "product"
    scale: float = 1.0  # effect = scale * (var1 * var2)

    def compute(self, df: pd.DataFrame) -> np.ndarray:
        if self.kind == "product":
            return self.scale * (df[self.var1].values * df[self.var2].values)
        else:
            raise ValueError(f"Unknown interaction kind: {self.kind}")


# =========================
# Modifier / baseline config
# =========================


class SeasonalitySpec(BaseModel):
    name: str = "seasonality"
    effect_scale: float = 200.0
    baseline: float = 0.0  # swing around the mean if zero
    amplitude: float = 0.2
    cycles: float = 2.0  # number of full sine cycles over the horizon
    noise_sigma: float = 0.0  # small realism for the proxy index
    disabled: bool = False

    def create(self, N: int, rng) -> Tuple[np.ndarray, np.ndarray]:
        t = np.linspace(0, 2 * np.pi * self.cycles, N)
        demand_index = self.baseline + self.amplitude * np.sin(t)

        if self.noise_sigma > 0:
            demand_index = demand_index + rng.normal(0, self.noise_sigma, size=N)

        effect = np.zeros(N)
        if not self.disabled:
            effect = self.effect_scale * demand_index

        return demand_index, effect


class CompetitorSpec(BaseModel):
    name: str = "competitor_discount"
    penalty_scale: float = 150.0  # subtract from conversions when event occurs
    probability: float = 0.1  # daily chance of event
    disabled: bool = False

    def create(self, N: int, rng) -> Tuple[np.ndarray, np.ndarray]:
        event = rng.binomial(1, self.probability, size=N)
        effect = np.zeros(N)
        if not self.disabled:
            effect = - self.penalty_scale * event
        return event, effect


class PriceIndexSpec(BaseModel):
    name: str = "price_index"
    penalty_scale: float = 250.0  # subtract penalty_scale * (price - 1.0)
    mu: float = 1.0
    sigma: float = 0.05
    disabled: bool = False

    def create(self, N: int, rng) -> Tuple[np.ndarray, np.ndarray]:
        index = rng.normal(self.mu, self.sigma, size=N)
        effect = np.zeros(N)
        if not self.disabled:
            effect = - self.penalty_scale * (index - 1.0)
        return index, effect


class BrandBaselineSpec(BaseModel):
    name: str = "brand_strength"
    base_low: float = 180.0
    base_high: float = 260.0
    trend_total: float = 20.0  # linear increase across horizon
    noise_sigma: float = 5.0
    disabled: bool = False

    def create(self, N: int, rng) -> Tuple[np.ndarray, np.ndarray]:
        if self.disabled:
            return np.zeros(N), np.zeros(N)
        base = rng.uniform(self.base_low, self.base_high, size=N)
        trend = np.linspace(0, self.trend_total, N)
        noise = rng.normal(0, self.noise_sigma, size=N)
        baseline_strength = base + trend + noise
        return np.arange(N), baseline_strength


class ModifierConfig(BaseModel):
    seasonality: SeasonalitySpec = SeasonalitySpec()
    competitor: CompetitorSpec = CompetitorSpec()
    price_index: PriceIndexSpec = PriceIndexSpec()
    brand: BrandBaselineSpec = BrandBaselineSpec()


# =========================
# Global synthetic config
# =========================


class SyntheticConfig(BaseModel):
    n: int = 500
    start_date: str = "2024-01-01"
    channels: list[ChannelConfig]
    interactions: list[InteractionSpec] = []
    modifiers: ModifierConfig = ModifierConfig()
    residual_noise_sigma: float = 100.0
    random_seed: int = 42


# =========================
# Data generator
# =========================


class DataGenerator(BaseModel):
    cfg: SyntheticConfig

    # runtime artifacts
    rng: np.random.Generator = Field(default_factory=lambda: np.random.default_rng(42))

    # output synthetic with everything non-latent
    # - drivers
    #   - spends
    #   - confounders
    # - conversions

    synthetic_df: Optional[pd.DataFrame] = None

    # internal state
    drivers: Optional[pd.DataFrame] = None  # drivers only: spends + confounders
    effects: Optional[pd.DataFrame] = None  # individual effects on conversions

    class Config:
        arbitrary_types_allowed = True

    @property
    def modifiers(self) -> list[SeasonalitySpec | CompetitorSpec | PriceIndexSpec | BrandBaselineSpec]:
        """List of all modifier specs."""
        return [
            self.cfg.modifiers.seasonality,
            self.cfg.modifiers.competitor,
            self.cfg.modifiers.price_index,
            self.cfg.modifiers.brand,
        ]

    def __init__(self, **data):
        super().__init__(**data)
        self.rng = np.random.default_rng(self.cfg.random_seed)

    # ---------- spends ----------
    def _create_spends(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        N = self.cfg.n
        channel_spends = {}
        effects = {}

        for ch in self.cfg.channels:
            low, high = ch.spend_range
            spend = self.rng.uniform(low, high, size=N)
            deterministic = ch.effect.apply(spend)
            noisy = deterministic + self.rng.normal(0, ch.noise_scale, size=N)

            # lag effect if requested (on effect, not spend)
            if ch.lag_days > 0:
                eff = np.roll(noisy, ch.lag_days)
                # pad the first entries with the first valid effect
                eff[: ch.lag_days] = eff[ch.lag_days]
            else:
                eff = noisy

            channel_spends[ch.name] = spend
            effects[ch.name] = eff

        # stash effects for later
        return pd.DataFrame(channel_spends), pd.DataFrame(effects)

    def _create_modifiers(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        N = self.cfg.n
        modifiers = {}
        effects = {}
        for cfg in self.modifiers:
            modifiers[cfg.name], effects[cfg.name] = cfg.create(N, rng=self.rng)

        return pd.DataFrame(modifiers), pd.DataFrame(effects)

    # ---------- assembly ----------
    def build_synthetic_df(self) -> pd.DataFrame:
        N = self.cfg.n
        synthetic_df = pd.DataFrame({"date": pd.date_range(self.cfg.start_date, periods=N, freq="D")})

        # spends + primary channel effects
        spends, spend_effects = self._create_spends()
        for c in spends.columns:
            synthetic_df[c] = spends[c]

        # sum channel effects
        channels_sum = np.zeros(N)
        for _, eff in spend_effects.items():
            channels_sum += eff.values

        # interactions (operate on spend columns by *name*)
        interaction_total = np.zeros(N)
        for inter in self.cfg.interactions:
            interaction_total += inter.compute(synthetic_df)

        # modifiers / confounders
        modifier_indices, modifier_effects = self._create_modifiers()
        modifier_effect = np.zeros(N)
        for _, eff in modifier_effects.items():
            modifier_effect += eff.values

        # add the modifier drivers to the synthetic dataframe to model as confounders
        synthetic_df = pd.concat([synthetic_df, modifier_indices], axis=1)

        # build conversions
        conv = (
            channels_sum
            + interaction_total
            + modifier_effect
            + self.rng.normal(0, self.cfg.residual_noise_sigma, size=N)
        )
        synthetic_df["conversions"] = conv

        # remove the disabled modifiers
        synthetic_df = synthetic_df.drop(columns=[m.name for m in self.modifiers if m.disabled])

        # set internal state
        self.synthetic_df = synthetic_df
        self.drivers = pd.concat([spends, modifier_indices], axis=1)
        self.effects = pd.concat([spend_effects, modifier_effects], axis=1)

        return synthetic_df

    # ---------- plotting ----------
    def plot_effect(
        self,
        spends: bool = True,
        confounders: bool = True,
        response_type: Literal["total", "individual"] = "total",
        scale: bool = False,
        figsize=(10, 10),
    ):
        if self.synthetic_df is None:
            raise RuntimeError("Call build_synthetic_df() first.")

        from sklearn.preprocessing import StandardScaler
        if scale:
            scaler = StandardScaler()

        cols_to_plot = []

        if spends:
            cols_to_plot.extend([c.name for c in self.cfg.channels])
        if confounders:
            for m in self.modifiers:
                if not m.disabled:
                    cols_to_plot.append(m.name)

        if not cols_to_plot:
            print("Nothing to plot. Set spends=True and/or confounders=True.")
            return

        n = len(cols_to_plot)
        ncols = 2
        nrows = (n + 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.ravel()

        # look for lags to shift the effect
        channel_to_lag_mapping = {c.name: c.lag_days for c in self.cfg.channels}

        # pick colors from colormap
        colors = cm.get_cmap("tab10", n)

        for i, (ax, col) in enumerate(zip(axes, cols_to_plot)):
            if response_type == "total":
                y = self.synthetic_df["conversions"].copy()
            else:
                y = self.effects[col].copy()
            x = self.drivers[col].copy()

            # handle scaling
            if scale:
                # scale the x-axis
                x = pd.Series(scaler.fit_transform(x.values.reshape(-1, 1)).flatten())
                y = pd.Series(scaler.fit_transform(y.values.reshape(-1, 1)).flatten())

            if channel_to_lag_mapping.get(col, 0) > 0:
                # if lagged, plot the lagged spend
                x = x.shift(channel_to_lag_mapping[col])
                ax.scatter(x, y, s=12, alpha=0.7, color=colors(i))
                ax.set_xlabel(f"{col} (*LAGGED*)")
            else:
                ax.scatter(x, y, s=12, alpha=0.7, color=colors(i))
                ax.set_xlabel(col)
            ax.set_ylabel("conversions")

        # tidy up empty axes
        for j in range(len(cols_to_plot), len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        fig.suptitle("Individual Contributions To Conversions", fontsize=16, y=1.02)
        return fig

    def plot_as_time_series(self):
        # Set up tall layout — adjust height to fit number of variables
        vars_to_plot = self.synthetic_df.columns.drop("date")  # everything except date
        n_vars = len(vars_to_plot)

        # pick colors from colormap
        colors = cm.get_cmap("tab10", n_vars)

        fig, axs = plt.subplots(3, 3, figsize=(10, 6), sharex=True)
        axs = axs.flatten()  # flatten to iterate easily
        df_indexed = self.synthetic_df.set_index("date")

        for i, (ax, col) in enumerate(zip(axs, vars_to_plot)):
            color = colors(i)
            df_indexed[col].plot(ax=ax, color=color, title=col)

            ymin, ymax = df_indexed[col].min(), df_indexed[col].max()
            pad = (ymax - ymin) * 0.25
            ax.set_ylim(ymin - pad, ymax + pad)

            ax.set_ylabel(col)

        plt.tight_layout()
        fig.suptitle("Synthetic Data Time Series", fontsize=16, y=1.02)

        return fig
