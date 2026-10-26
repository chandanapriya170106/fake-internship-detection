"""A tiny educational blockchain for verified employers / offer letters.

Each block stores a SHA-256 hash of the previous block, so editing any past
record breaks the chain (tamper-proof). Run:  python -m blockchain.ledger
"""
import hashlib
import json
import os
import time

LEDGER_FILE = "blockchain/ledger.json"
DIFFICULTY = 3  # hash must start with this many zeros (proof of work)


def sha256(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def hash_document(text: str) -> str:
    """Fingerprint of an offer letter / document."""
    return sha256(text.strip())


class Blockchain:
    def __init__(self, path=LEDGER_FILE):
        self.path = path
        self.chain = []
        if os.path.exists(path):
            with open(path) as f:
                self.chain = json.load(f)
        else:
            self._add_block({"type": "genesis"})

    @staticmethod
    def _block_hash(b):
        body = {k: b[k] for k in ("index", "timestamp", "data", "prev_hash", "nonce")}
        return sha256(json.dumps(body, sort_keys=True))

    def _add_block(self, data):
        block = {
            "index": len(self.chain),
            "timestamp": time.time(),
            "data": data,
            "prev_hash": self.chain[-1]["hash"] if self.chain else "0" * 64,
            "nonce": 0,
        }
        while not self._block_hash(block).startswith("0" * DIFFICULTY):
            block["nonce"] += 1
        block["hash"] = self._block_hash(block)
        self.chain.append(block)
        self.save()
        return block

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.chain, f, indent=2)

    # ---- employer registry ----
    def register_employer(self, company: str, website: str, registration_no: str):
        return self._add_block({"type": "employer", "company": company.lower().strip(),
                                "website": website.lower().strip(), "reg_no": registration_no})

    def is_verified_employer(self, company: str) -> bool:
        return any(b["data"].get("type") == "employer" and b["data"]["company"] == company.lower().strip()
                   for b in self.chain)

    # ---- offer letters ----
    def issue_offer(self, company: str, offer_text: str):
        if not self.is_verified_employer(company):
            raise ValueError("Employer is not registered on the blockchain")
        return self._add_block({"type": "offer", "company": company.lower().strip(),
                                "doc_hash": hash_document(offer_text)})

    def verify_offer(self, offer_text: str) -> bool:
        h = hash_document(offer_text)
        return any(b["data"].get("type") == "offer" and b["data"]["doc_hash"] == h for b in self.chain)

    # ---- integrity ----
    def is_valid(self) -> bool:
        for i, b in enumerate(self.chain):
            if b["hash"] != self._block_hash(b):
                return False
            if i > 0 and b["prev_hash"] != self.chain[i - 1]["hash"]:
                return False
        return True


if __name__ == "__main__":
    bc = Blockchain("blockchain/demo_ledger.json")
    if not bc.is_verified_employer("Zoho Corp"):
        bc.register_employer("Zoho Corp", "https://www.zoho.com", "U72200TN1996PTC")
    letter = "Offer: Priya S selected as Data Analyst Intern at Zoho Corp, 3 months."
    bc.issue_offer("Zoho Corp", letter)
    print("Genuine offer verified :", bc.verify_offer(letter))
    print("Edited offer verified  :", bc.verify_offer(letter + " Pay Rs 999 fee."))
    print("Chain valid            :", bc.is_valid())
    bc.chain[1]["data"]["company"] = "hacker inc"   # tamper
    print("Chain valid after edit :", bc.is_valid())
