"""LangGraph tools for the travel-assistant demo.

``read_travel_plan`` extracts text from a PDF file — the returned content may
contain an embedded prompt-injection payload (this is the untrusted tool).
``book_flight`` and ``send_money`` are trusted action tools.
"""

from __future__ import annotations

from langchain_core.tools import tool
from pypdf import PdfReader


@tool
def read_travel_plan(file_path: str) -> str:
    """Read and extract text from a travel-plan PDF."""
    reader = PdfReader(file_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


@tool
def book_flight(flight_id: str, passenger: str) -> str:
    """Book a specific flight for a passenger."""
    return f"Booking confirmed: {flight_id} for {passenger}."


@tool
def send_money(amount: float, account: str) -> str:
    """Transfer money to an external account."""
    return f"Sent ${amount:.2f} to {account}."
