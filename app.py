"""Streamlit app:  streamlit run app.py"""
import os
import streamlit as st

from blockchain.ledger import Blockchain
from src.predict import predict_posting

st.set_page_config(page_title="Fake Internship Detector", page_icon="🛡️")
st.title("🛡️ Fake Internship Detector")

if not os.path.exists("models/fake_internship_model.joblib"):
    st.error("Model not found. Run `python -m src.train` first.")
    st.stop()

tab1, tab2 = st.tabs(["ML Detection", "Blockchain Verification"])

with tab1:
    st.write("Paste an internship posting or offer message below.")
    text = st.text_area("Posting text", height=220,
                        placeholder="Urgent!!! Pay Rs 999 registration fee. No interview. Mail hr@gmail.com ...")
    if st.button("Analyse", type="primary"):
        if not text.strip():
            st.warning("Please paste some text.")
        else:
            r = predict_posting(text)
            if r["label"] == "Fake":
                st.error(f"⚠️ Likely FAKE  ({r['confidence']:.0%} confidence)")
            else:
                st.success(f"✅ Likely GENUINE  ({r['confidence']:.0%} confidence)")
            st.progress(r["p_fake"], text=f"Fraud probability: {r['p_fake']:.0%}")
            if r["red_flags"]:
                st.subheader("Red flags found")
                for f in r["red_flags"]:
                    st.write(f"• {f}")
            st.caption(f"Model: {r['model']}. Always double-check on the company's official website.")

with tab2:
    bc = Blockchain()
    st.write(f"Ledger blocks: **{len(bc.chain)}**  |  Chain valid: **{bc.is_valid()}**")

    st.subheader("Register employer (demo)")
    c1, c2, c3 = st.columns(3)
    name = c1.text_input("Company")
    site = c2.text_input("Website")
    reg = c3.text_input("Registration no.")
    if st.button("Register on chain"):
        if name and site and reg:
            bc.register_employer(name, site, reg)
            st.success("Employer registered.")
        else:
            st.warning("Fill all fields.")

    st.subheader("Issue offer letter hash (verified employers only)")
    ic = st.text_input("Employer name", key="ic")
    letter = st.text_area("Offer letter text", key="letter", height=120)
    if st.button("Issue offer"):
        try:
            b = bc.issue_offer(ic, letter)
            st.success(f"Stored hash {b['data']['doc_hash'][:16]}... in block #{b['index']}")
        except ValueError as e:
            st.error(str(e))

    st.subheader("Verify an offer letter")
    check = st.text_area("Paste the offer letter you received", key="check", height=120)
    if st.button("Verify"):
        if bc.verify_offer(check):
            st.success("✅ This offer letter exists on the blockchain (genuine).")
        else:
            st.error("❌ Not found. It may be fake or has been altered.")
