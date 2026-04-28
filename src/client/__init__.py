"""Flower client for federated ADMET training."""
from src.client.flower_client import ADMETClient, make_client_fn

__all__ = ["ADMETClient", "make_client_fn"]
