import streamlit as st
from datetime import date
from io import StringIO

st.set_page_config(page_title="Dynamic SEO Generator", layout="wide")
st.title("Dynamic SEO Engine (Separate System)")

st.markdown(
    "This tool simulates the **Dynamic SEO system** that listens to search intent and generates optimized landing content. "
    "It uses prompt templates and your inputs to generate consistent, compliant SEO blocks."
)

with st.sidebar:
    st.header("Brand & Product Facts (Source of Truth)")
    brand = st.text_input("Brand / Card Name", value="NeuCard")
    usp_list = st.text_area("Core Benefits (one per line)", value="5% back on Tata brands\nFuel surcharge waiver\nMilestone bonus NeuCoins\nZero joining fee (limited offer)")
    fees = st.text_area("Fees & Charges (bullets)", value="Joining fee: ₹0 (limited offer)\nAnnual fee: ₹499\nAPR: 3.1% per month (37.2% p.a.)")
    eligibility = st.text_area("Eligibility (bullets)", value="Age 21–60\nValid PAN & address proof\nStable income or FD for secured card")
    last_updated = date.today().isoformat()

st.subheader("1) Enter Search Intent")
query = st.text_input("Search query / intent", value="best credit card in India for groceries")
audience = st.selectbox("Audience", ["General", "Students", "Travelers", "Online Shoppers"], index=3)
tone = st.selectbox("Tone", ["Neutral", "Helpful", "Trust-building", "Direct"], index=2)

def gen_title(query, brand):
    base = query.strip().capitalize()
    t = f"{base} — {brand}"
    return t[:60]

def gen_meta(query, brand):
    meta = f"{brand}: Rewards tailored for '{query}'. Transparent fees, fast approval, and NeuCoins. Check eligibility & apply in minutes."
    return meta[:155]

def gen_h1(query, brand):
    return f"{brand}: {query.title()}"[:70]

def gen_sections(query, usp_list, fees, eligibility, audience, tone, last_updated, brand):
    usps = [u.strip() for u in usp_list.splitlines() if u.strip()]
    faqs = [
        (f"Is {brand} good for {query}?", f"Yes. {brand} offers benefits aligned with '{query}', including: " + ", ".join(usps[:3]) + "."),
        (f"How do I apply for {brand}?", f"Check your eligibility online and complete KYC. Instant decision for many users; secured option via FD is also available.")
    ]
    return usps, faqs

if st.button("Generate SEO Pack"):
    usps, faqs = gen_sections(query, usp_list, fees, eligibility, audience, tone, last_updated, brand)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("SEO Title")
        title = gen_title(query, brand)
        st.write(title)
        st.subheader("Meta Description")
        meta = gen_meta(query, brand)
        st.write(meta)
        st.subheader("H1")
        h1 = gen_h1(query, brand)
        st.write(h1)
        st.subheader("Benefit Bullets")
        st.write(usps)

    with col2:
        st.subheader("FAQs (for FAQPage Schema)")
        for i, (q, a) in enumerate(faqs, 1):
            st.markdown(f"**Q{i}. {q}**")
            st.write(a)

        st.subheader("Product Facts")
        st.text(fees)
        st.text(eligibility)

    # JSON-LD
    json_ld = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": [
            {"@type": "Question", "name": q, "acceptedAnswer": {"@type": "Answer", "text": a}}
            for q, a in faqs
        ]
    }

    st.subheader("JSON-LD (FAQ)")
    st.code(str(json_ld), language="json")

    # Markdown export
    md = f"# {h1}\n_Last updated: {last_updated}_\n\n"
    md += f"**Title (<=60):** {title}\n"
    md += f"**Meta (<=155):** {meta}\n\n"
    md += f"## Why {brand}\n" + "\n".join([f"- {u}" for u in usps]) + "\n\n"
    md += "## Fees & Charges\n" + fees + "\n\n"
    md += "## Eligibility\n" + eligibility + "\n\n"
    md += "## FAQs\n" + "\n".join([f"**Q:** {q}\n\n**A:** {a}\n" for q, a in faqs])

    st.subheader("Export")
    buf = StringIO()
    buf.write(md)
    st.download_button("Download Markdown", data=buf.getvalue(), file_name="seo_pack.md", mime="text/markdown")

st.markdown("---")
st.caption("Tip: Pair this with your CMS to publish quickly. For production, add human review + change logs.")
