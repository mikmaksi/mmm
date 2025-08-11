from pathlib import Path

import numpy as np

from data_generator import (
    BrandBaselineSpec,
    ChannelConfig,
    CompetitorSpec,
    DataGenerator,
    EffectSpec,
    InteractionSpec,
    ModifierConfig,
    PriceIndexSpec,
    SeasonalitySpec,
    SyntheticConfig,
)

# %% -
# configure the data generation
modifier_config = ModifierConfig(
    seasonality=SeasonalitySpec(effect_scale=200.0, baseline=0, amplitude=0.2, cycles=2.0),
    competitor=CompetitorSpec(penalty_scale=150.0, probability=0.15),
    price_index=PriceIndexSpec(penalty_scale=250.0, mu=1.0, sigma=0.05),
    brand=BrandBaselineSpec(base_low=180, base_high=260, trend_total=20.0, noise_sigma=5.0, disabled=True),
)
cfg = SyntheticConfig(
    n=500,
    start_date="2024-01-01",
    random_seed=42,
    channels=[
        ChannelConfig(
            name="spend_search_ads",
            spend_range=(0, 5000),
            effect=EffectSpec(kind="exp_sat", scale=1200, k=0.001),
            noise_scale=8.0,
            lag_days=0,
        ),
        # moderate power effect
        ChannelConfig(
            name="spend_social_media",
            spend_range=(0, 2600),
            effect=EffectSpec(kind="sqrt", scale=30.0),
            noise_scale=30.0,
            lag_days=0,
        ),
        # near-linear effect with 1-day lag
        ChannelConfig(
            name="spend_tv",
            spend_range=(0, 10000),
            effect=EffectSpec(kind="linear", a=0.07, b=0.0),
            noise_scale=12.0,
            lag_days=1,
        ),
        # sigmoid effect
        ChannelConfig(
            name="spend_email",
            spend_range=(0, 300),
            effect=EffectSpec(kind="sigmoid", scale=1100.0, k=0.1, x0=0.0),
            noise_scale=32.0,
            lag_days=0,
        ),
        # weak linear effect but has an interaction with search ads
        ChannelConfig(
            name="spend_affiliates",
            spend_range=(0, 2000),
            effect=EffectSpec(kind="linear", a=0.15, b=500.0),
            noise_scale=8.0,
            lag_days=0,
        ),
    ],
    interactions=[
        # search x affiliates interaction (scaled to keep magnitudes sane)
        InteractionSpec(name="search_affiliates", var1="spend_search_ads", var2="spend_affiliates", scale=0.00003),
    ],
    modifiers=modifier_config,
    residual_noise_sigma=70.0,
)

# %% -
# build the data
gen = DataGenerator(cfg=cfg)
synthetic_data = gen.build_synthetic_df()

# %% -
# save data
OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
synthetic_data.to_csv(OUT_DIR / "synthetic_data.csv", index=False)

# %% -
# save plots
PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)
scale = False
fig = gen.plot_effect(spends=True, confounders=False, response_type="total", scale=scale, figsize=(8, 8))
fig.savefig(PLOT_DIR / "total_effect.png")
fig = gen.plot_effect(spends=True, confounders=False, response_type="individual", scale=scale, figsize=(8, 8))
fig.savefig(PLOT_DIR / "individual_effect.png")
fig = gen.plot_as_time_series()
fig.savefig(PLOT_DIR / "synthetic_data_timeseries.png", dpi=300, bbox_inches="tight")

# %% -
gen.synthetic_df.head()
gen.drivers
gen.effects

# %% -
# SNR sanity check
noise_scales = {ch.name: ch.noise_scale for ch in gen.cfg.channels}
for ch in ["spend_search_ads", "spend_social_media", "spend_tv", "spend_email", "spend_affiliates"]:
    v = np.var(gen.effects[ch])
    snr = v / (noise_scales[ch] ** 2)
    print(ch, round(snr, 2))
