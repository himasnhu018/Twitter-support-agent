import streamlit as st
from scripts.agent import AmazonSupportAgent


# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------

st.set_page_config(
    page_title="AmazonHelp Support Agent",
    page_icon="🤖",
    layout="wide",
)


# ------------------------------------------------------------
# STYLING
# ------------------------------------------------------------

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #666;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .decision {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        margin-top: 1rem;
    }

    .case {
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        margin-bottom: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# LOAD AGENT
# ------------------------------------------------------------

@st.cache_resource
def load_agent():
    return AmazonSupportAgent(k=5)


try:
    agent = load_agent()
except Exception as e:
    st.error(f"Could not load the support agent: {e}")
    st.stop()


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

st.markdown(
    '<div class="main-title">🤖 AmazonHelp AI Support Agent</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    RAG-powered customer support assistant using historical AmazonHelp
    conversations.
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# INPUT
# ------------------------------------------------------------

message = st.text_area(
    "Customer message",
    placeholder="Example: My package was supposed to arrive yesterday but it still hasn't arrived.",
    height=130,
)


run_button = st.button(
    "Analyze Customer Message",
    type="primary",
    use_container_width=True,
)


# ------------------------------------------------------------
# RUN PIPELINE
# ------------------------------------------------------------

if run_button:

    if not message.strip():
        st.warning("Please enter a customer message.")
        st.stop()

    with st.spinner("Running support agent..."):
        try:
            result = agent.run(message.strip())
        except Exception as e:
            st.error(f"Agent error: {e}")
            st.stop()

    st.divider()

    # --------------------------------------------------------
    # INTENT + DECISION
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Intent")
        st.metric(
            "Predicted intent",
            result["intent"],
        )
        st.caption(
            f"Confidence: {result['intent_confidence']:.2f}"
        )
        st.write(result["intent_reason"])

    with col2:
        st.subheader("Decision")

        if result["decision"] == "ESCALATE":
            st.error("🚨 ESCALATE")
        else:
            st.success("✅ AUTO_HANDLE")

        st.caption(
            f"Confidence: {result['escalation_confidence']:.2f}"
        )

    with col3:
        st.subheader("Response confidence")
        st.metric(
            "Confidence",
            f"{result['response_confidence']:.2f}",
        )

    # --------------------------------------------------------
    # ESCALATION REASON
    # --------------------------------------------------------

    st.subheader("Escalation reasoning")

    st.info(result["escalation_reason"])

    # --------------------------------------------------------
    # DRAFT RESPONSE
    # --------------------------------------------------------

    st.subheader("Draft customer response")

    st.text_area(
        "Generated response",
        value=result["reply"],
        height=140,
        label_visibility="collapsed",
    )

    st.caption(
        f"Grounding: {result['grounding']}"
    )

    # --------------------------------------------------------
    # RETRIEVED CASES
    # --------------------------------------------------------

    st.divider()

    st.subheader("Retrieved historical support cases")

    cases = result.get("retrieved_cases", [])

    if not cases:
        st.warning("No historical cases were retrieved.")
    else:
        for case in cases:

            with st.expander(
                f"Rank {case['rank']}  •  "
                f"Similarity: {case['similarity']:.4f}"
            ):

                st.markdown("**Historical customer message**")
                st.write(case["customer_text"])

                st.markdown("**Historical AmazonHelp response**")
                st.write(case["agent_response"])


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

with st.sidebar:

    st.header("About")

    st.write(
        """
        This prototype uses:

        - Gemini for intent classification
        - FAISS for semantic retrieval
        - Historical AmazonHelp support cases
        - Gemini for grounded response generation
        - Gemini for escalation decisions
        """
    )

    st.divider()

    st.subheader("Pipeline")

    st.write(
        """
        Customer message
        ↓

        Intent classification
        ↓

        Historical case retrieval
        ↓

        Grounded response generation
        ↓

        Escalation decision
        """
    )

    st.divider()

    st.caption(
        "Historical responses are treated as evidence, "
        "not authoritative current policy."
    )