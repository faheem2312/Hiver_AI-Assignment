"""
Intent Taxonomy Definition for @AmazonHelp AI Support Agent
============================================================
Single source of truth for intent classification across the entire pipeline.
Derived empirically from clustering and domain analysis of Amazon Twitter customer support threads.
"""

from enum import Enum
from typing import List, Dict, Any


class Intent(str, Enum):
    ORDER_TRACKING_DELAY = "ORDER_TRACKING_DELAY"
    REFUND_RETURN_INQUIRY = "REFUND_RETURN_INQUIRY"
    DAMAGED_WRONG_ITEM = "DAMAGED_WRONG_ITEM"
    PRIME_MEMBERSHIP_BILLING = "PRIME_MEMBERSHIP_BILLING"
    ACCOUNT_LOGIN_SECURITY = "ACCOUNT_LOGIN_SECURITY"
    PRODUCT_TECH_SUPPORT = "PRODUCT_TECH_SUPPORT"
    ESCALATION_HIGH_RISK = "ESCALATION_HIGH_RISK"
    OTHER_GENERAL = "OTHER_GENERAL"

    @property
    def description(self) -> str:
        """Clear, authoritative definition used in classifier prompts and rubrics."""
        descriptions = {
            Intent.ORDER_TRACKING_DELAY: (
                "Customer inquiring about order status, tracking updates, late or missing deliveries, "
                "packages marked 'delivered' but not received, or carrier transit delays."
            ),
            Intent.REFUND_RETURN_INQUIRY: (
                "Questions regarding how to return an item, return eligibility windows (e.g. 30 days), "
                "drop-off locations (UPS, Kohl's), or refund processing timelines to original payment method."
            ),
            Intent.DAMAGED_WRONG_ITEM: (
                "Reports of receiving broken, defective, or incorrect products, shattered glass, "
                "torn packaging, or missing components inside the box."
            ),
            Intent.PRIME_MEMBERSHIP_BILLING: (
                "Inquiries about Prime subscription charges, unexpected auto-renewal fees, cancellation, "
                "prorated refunds for unused Prime, or Prime video / delivery perk questions."
            ),
            Intent.ACCOUNT_LOGIN_SECURITY: (
                "Issues with Two-Factor / OTP authentication codes, locked accounts, password reset "
                "failures, or security alerts regarding unauthorized account access."
            ),
            Intent.PRODUCT_TECH_SUPPORT: (
                "Technical troubleshooting for Amazon devices including Echo / Alexa (blinking rings), "
                "Fire TV Stick (frozen screens, remote unpairing), or Kindle e-readers."
            ),
            Intent.ESCALATION_HIGH_RISK: (
                "Severe customer friction requiring human intervention: property damage by delivery drivers, "
                "high-value stolen items (> $100), explicit legal or attorney threats, police reports, or extreme abuse."
            ),
            Intent.OTHER_GENERAL: (
                "Vague greetings, general feedback, website bug reports, or queries that do not fit into any specific category above."
            ),
        }
        return descriptions.get(self, "Unclassified intent.")

    @property
    def default_action(self) -> str:
        """Baseline routing action recommendation ('auto' or 'escalate')."""
        if self in [Intent.ESCALATION_HIGH_RISK, Intent.ACCOUNT_LOGIN_SECURITY]:
            return "escalate"
        return "auto"

    @property
    def is_high_risk(self) -> bool:
        """Flags whether this intent carries strict human-oversight policy."""
        return self == Intent.ESCALATION_HIGH_RISK

    @classmethod
    def list_all(cls) -> List[str]:
        """Returns list of all valid intent string keys."""
        return [intent.value for intent in cls]

    @classmethod
    def get_prompt_taxonomy(cls) -> str:
        """
        Formats the taxonomy as a clean, structured string for LLM system prompts.
        """
        lines = []
        for intent in cls:
            lines.append(f"- **{intent.value}**: {intent.description}")
        return "\n".join(lines)


if __name__ == "__main__":
    print("=== Amazon Customer Support Intent Taxonomy ===")
    print(Intent.get_prompt_taxonomy())
    print("\nTotal Registered Intents:", len(Intent.list_all()))
