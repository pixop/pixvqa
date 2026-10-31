"""Optional subprocess adapter for the official ColorVideoVDP CLI."""

PINNED_UPSTREAM = "https://github.com/gfxdisp/ColorVideoVDP.git"
PINNED_COMMIT = "166f5ae95ecbd841ae727984cd237de213875399"
PINNED_VERSION = "0.5.7"
PINNED_TORCH_VERSION = "2.11.0+cu128"
PINNED_TORCHVISION_VERSION = "0.26.0+cu128"

# Fields copied from pycvvdp/vvdp_data/display_models.json at PINNED_COMMIT.
DISPLAY_MODELS = {
    "standard_fhd": {
        "interpretation": "sdr",
        "colorspace": "sRGB (package default; colorspace key omitted in JSON)",
        "resolution": [1920, 1080],
        "viewing_distance_meters": 0.6,
        "diagonal_size_inches": 24,
        "max_luminance": 200,
        "contrast": 1000,
        "E_ambient": 250,
    },
    "standard_4k": {
        "interpretation": "sdr",
        "colorspace": "sRGB (package default; colorspace key omitted in JSON)",
        "resolution": [3840, 2160],
        "viewing_distance_meters": 0.7472,
        "diagonal_size_inches": 30,
        "max_luminance": 200,
        "contrast": 1000,
        "E_ambient": 250,
    },
    "standard_hdr_pq": {
        "interpretation": "pq",
        "colorspace": "BT.2020-PQ",
        "resolution": [3840, 2160],
        "viewing_distance_meters": 0.7472,
        "diagonal_size_inches": 30,
        "max_luminance": 1500,
        "contrast": 1000000,
        "E_ambient": 10,
    },
    "standard_hdr_hlg": {
        "interpretation": "hlg",
        "colorspace": "BT.2020-HLG",
        "resolution": [3840, 2160],
        "viewing_distance_meters": 0.7472,
        "diagonal_size_inches": 30,
        "max_luminance": 1500,
        "contrast": 1000000,
        "E_ambient": 10,
    },
}

SDR_DISPLAYS = ("standard_fhd", "standard_4k")
PQ_DISPLAYS = ("standard_hdr_pq",)
HLG_DISPLAYS = ("standard_hdr_hlg",)
