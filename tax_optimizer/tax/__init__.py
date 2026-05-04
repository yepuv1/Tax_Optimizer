"""Tax engine: regime-aware federal income tax + IRMAA surcharges."""

from .federal import (
    amount_to_fill_bracket,
    federal_tax,
    social_security_taxable,
)
from .irmaa import MEDICARE_ELIGIBLE_AGE, irmaa_annual_surcharge
from .regimes import (
    PRE_TCJA_2017,
    SUNSET_2026,
    TCJA_EXTENDED,
    TaxRegime,
)

__all__ = [
    "TaxRegime",
    "TCJA_EXTENDED",
    "PRE_TCJA_2017",
    "SUNSET_2026",
    "federal_tax",
    "social_security_taxable",
    "amount_to_fill_bracket",
    "irmaa_annual_surcharge",
    "MEDICARE_ELIGIBLE_AGE",
]
