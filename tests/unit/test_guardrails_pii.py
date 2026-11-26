"""Unit tests for PII detection and masking."""

from __future__ import annotations

from rag_engine.guardrails.pii import luhn_checksum_valid, mask_pii


def test_luhn_checksum_valid() -> None:
    # Valid Visa card numbers (standard test card numbers)
    assert luhn_checksum_valid("4532015112830366")
    # Invalid card number (modified last digit)
    assert not luhn_checksum_valid("4532015112830367")
    # Too short
    assert not luhn_checksum_valid("12345")


def test_mask_email_addresses() -> None:
    text = "Please reach out to alice.smith@example.com or bob-admin@corp.org for help."
    result = mask_pii(text)

    assert result.has_pii
    assert len(result.entities) == 2
    assert "[EMAIL_ADDRESS]" in result.text
    assert "alice.smith@example.com" not in result.text
    assert "bob-admin@corp.org" not in result.text
    assert result.entities[0].placeholder == "[EMAIL_ADDRESS]"


def test_mask_phone_numbers() -> None:
    text = "Call support at (555) 123-4567 or emergency line +1-800-555-0199."
    result = mask_pii(text)

    assert result.has_pii
    assert len(result.entities) >= 1
    assert "[PHONE_NUMBER]" in result.text
    assert "(555) 123-4567" not in result.text


def test_mask_ssn() -> None:
    text = "The employee SSN on file is 123-45-6789."
    result = mask_pii(text)

    assert result.has_pii
    assert "[SSN]" in result.text
    assert "123-45-6789" not in result.text
    assert result.entities[0].entity_type == "ssn"


def test_mask_credit_card() -> None:
    # Formatted valid card number
    text = "Payment details: 4532-0151-1283-0366 expires 12/28."
    result = mask_pii(text)

    assert result.has_pii
    assert "[CREDIT_CARD]" in result.text
    assert "4532-0151-1283-0366" not in result.text
    assert any(e.entity_type == "credit_card" for e in result.entities)


def test_invalid_credit_card_is_not_masked() -> None:
    # 16 digits that fail Luhn check (e.g. an order ID)
    text = "Tracking serial: 1111-2222-3333-4445."
    result = mask_pii(text)
    # Shouldn't be tagged as credit_card (might match phone or not at all)
    assert not any(e.entity_type == "credit_card" for e in result.entities)


def test_mask_api_keys() -> None:
    text = (
        "My OpenAI key is sk-proj-abcdef1234567890abcdef123456 and "
        "GitHub token is ghp_1234567890abcdefghijklmnopqrstuvwxyz."
    )
    result = mask_pii(text)

    assert result.has_pii
    assert "[API_KEY]" in result.text
    assert "sk-proj-" not in result.text
    assert "ghp_" not in result.text


def test_multiple_pii_entities_in_sentence() -> None:
    text = "User john@acme.com with phone (555) 000-1111 submitted SSN 987-65-4321."
    result = mask_pii(text)

    assert result.has_pii
    assert "[EMAIL_ADDRESS]" in result.text
    assert "[PHONE_NUMBER]" in result.text
    assert "[SSN]" in result.text
    assert len(result.entities) == 3


def test_no_pii_present() -> None:
    text = "What is the policy for expense reimbursement?"
    result = mask_pii(text)

    assert not result.has_pii
    assert result.text == text
    assert len(result.entities) == 0
