"""Layer I arithmetic verification, including round trips."""
from __future__ import annotations

import pytest

from aegis.checksums import (
    iban_checksum,
    iban_valid,
    luhn_check_digit,
    luhn_valid,
    make_iban,
    ssn_plausible,
)


@pytest.mark.parametrize("number", [
    "4532015112830366", "4111111111111111", "5500005555555559", "378282246310005",
])
def test_luhn_accepts_known_valid(number):
    assert luhn_valid(number)


@pytest.mark.parametrize("number", ["4532015112830367", "4111111111111112", "1234567890123"])
def test_luhn_rejects_corrupted(number):
    assert not luhn_valid(number)


def test_luhn_check_digit_round_trip():
    body = "453201511283036"
    assert luhn_valid(body + luhn_check_digit(body))


def test_luhn_ignores_separators():
    assert luhn_valid("4532 0151 1283 0366")
    assert luhn_valid("4532-0151-1283-0366")


@pytest.mark.parametrize("iban", [
    "DE89370400440532013000", "GB82WEST12345698765432", "FR1420041010050500013M02606",
])
def test_iban_accepts_known_valid(iban):
    assert iban_valid(iban)


def test_iban_rejects_wrong_checksum():
    assert not iban_valid("DE88370400440532013000")


@pytest.mark.parametrize("country,bban", [
    ("DE", "370400440532013000"), ("NL", "ABNA0417164300"), ("ES", "21000418450200051332"),
])
def test_make_iban_round_trip(country, bban):
    built = make_iban(country, bban)
    assert iban_valid(built)
    assert built[:2] == country
    assert built[2:4] == iban_checksum(country, bban)


def test_ssn_rules():
    assert ssn_plausible("406-44-8691")
    assert not ssn_plausible("000-44-8691")   # area 000
    assert not ssn_plausible("666-44-8691")   # area 666
    assert not ssn_plausible("900-44-8691")   # area >= 900
    assert not ssn_plausible("406-00-8691")   # group 00
    assert not ssn_plausible("406-44-0000")   # serial 0000
