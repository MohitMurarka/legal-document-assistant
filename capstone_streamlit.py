# ============================================================
# capstone_streamlit.py — Legal Document Assistant
# Run: streamlit run capstone_streamlit.py
# ============================================================

import streamlit as st
import uuid
import os
from datetime import date, timedelta
from dotenv import load_dotenv
from typing import TypedDict, List

from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import chromadb

load_dotenv()

st.set_page_config(
    page_title="Legal Document Assistant", page_icon="⚖️", layout="centered"
)
st.title("⚖️ Legal Document Assistant")
st.caption("AI-powered legal research for paralegals and junior lawyers")

# ============================================================
# KNOWLEDGE BASE
# ============================================================
DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "Contract Formation and Elements",
        "text": """A valid contract requires four essential elements: offer, acceptance, consideration, and mutual assent (also called meeting of the minds). Without all four, a court may find the agreement unenforceable. An offer is a clear proposal by one party (the offeror) to another (the offeree) that specifies the essential terms and demonstrates a willingness to be bound. An offer must be distinguished from a mere invitation to negotiate. Acceptance must be unconditional and mirror the terms of the offer exactly. Under the mirror image rule, any attempt to accept with different terms constitutes a counteroffer, which terminates the original offer. Consideration is the bargained-for exchange — something of legal value given by each party. It can be a promise, an act, or a forbearance. Past consideration is not valid. Mutual assent means both parties genuinely agreed to the same terms. Contracts formed under duress, undue influence, misrepresentation, or mistake may be voidable. Additional requirements include capacity and legality. Written contracts are required for real estate transactions, agreements lasting more than one year, and sale of goods over $500 under the Statute of Frauds.""",
    },
    {
        "id": "doc_002",
        "topic": "Non-Disclosure Agreements (NDAs)",
        "text": """A Non-Disclosure Agreement (NDA), also called a confidentiality agreement, is a legally binding contract that restricts one or more parties from disclosing specified confidential information to third parties. NDAs are commonly used in employment, business partnerships, mergers and acquisitions, and intellectual property licensing. There are two types: unilateral NDAs (one party discloses, the other agrees not to share) and mutual NDAs (both parties exchange confidential information and both are bound). Key clauses include: Definition of Confidential Information, Exclusions (public domain info, independently developed info), Duration, and Permitted Disclosures. Breach of an NDA can result in injunctive relief, monetary damages, and disgorgement of profits. NDAs cannot protect information that is general knowledge or prevent employees from reporting illegal activity to government regulators.""",
    },
    {
        "id": "doc_003",
        "topic": "Employment Termination Law",
        "text": """Employment termination law governs when and how an employer may end an employment relationship. In most U.S. states, employment is at-will by default. Wrongful termination occurs when an employee is fired for an illegal reason, including termination based on protected characteristics under Title VII, the ADA, and the ADEA; retaliation for filing a workers' compensation claim; retaliation for whistleblowing; and termination that violates an implied or express employment contract. The WARN Act requires employers with 100 or more employees to provide 60 days written notice before a mass layoff or plant closing. Severance pay is not legally required under federal law unless promised. Employees over 40 must be given 21 days to consider a release and 7 days to revoke after signing, under the Older Workers Benefit Protection Act. Final paychecks must be issued on specific timelines that vary by state.""",
    },
    {
        "id": "doc_004",
        "topic": "Intellectual Property — Copyright Basics",
        "text": """Copyright is a form of intellectual property protection that grants the creator of an original work exclusive rights to reproduce, distribute, display, perform, and create derivative works. Copyright protection attaches automatically upon creation and fixation in a tangible medium. Registration with the U.S. Copyright Office provides significant legal advantages including the right to sue for infringement and eligibility for statutory damages up to $150,000 per work for willful infringement. Copyright protects original expression, not ideas, facts, procedures, or systems. For works created after January 1, 1978, copyright lasts for the life of the author plus 70 years. For works made for hire, copyright lasts 95 years from publication or 120 years from creation, whichever is shorter. Fair use is a defense that permits limited use without permission based on four factors: purpose, nature of work, amount used, and market effect.""",
    },
    {
        "id": "doc_005",
        "topic": "Intellectual Property — Trademark Law",
        "text": """A trademark is any word, name, symbol, logo, slogan, color, or combination used to identify the source of goods or services. Trademark rights in the U.S. arise from use in commerce, not registration alone. Federal registration with the USPTO provides a presumption of ownership and nationwide constructive notice. Trademarks are classified by distinctiveness: fanciful marks and arbitrary marks receive the strongest protection. Suggestive marks also qualify. Descriptive marks require acquired distinctiveness. Generic terms can never be registered. Trademark infringement occurs when a party uses a mark likely to cause consumer confusion. Trademark dilution protects famous marks from uses that blur or tarnish the mark's reputation even without consumer confusion. Trademark registration must be renewed with a Section 8 Declaration filed between the 5th and 6th years and every 10 years thereafter.""",
    },
    {
        "id": "doc_006",
        "topic": "Civil Litigation Process",
        "text": """Civil litigation is the process by which private parties resolve legal disputes through the court system. The process begins with pre-litigation investigation and litigation holds to prevent spoliation. The plaintiff files a complaint with the appropriate court and the defendant must be served. In federal court, the defendant has 21 days to respond by filing an answer or pre-answer motion. Discovery includes interrogatories, requests for production, depositions, and requests for admissions. Summary judgment can be sought after discovery. If no summary judgment is granted, the case proceeds to trial — jury or bench. A notice of appeal must be filed within 30 days of final judgment in federal court.""",
    },
    {
        "id": "doc_007",
        "topic": "Filing Deadlines and Statutes of Limitations",
        "text": """A statute of limitations is the deadline by which a lawsuit must be filed. If a plaintiff fails to file within the limitations period, the claim is permanently barred regardless of its merits. Common statutes of limitations include: personal injury (2–3 years in most states), breach of written contract (4–6 years), breach of oral contract (2–4 years), fraud (3–6 years from discovery), defamation (1–3 years), and medical malpractice (2–3 years). The clock typically begins when the cause of action accrues. The discovery rule may toll the statute until the plaintiff knew or should have known of the injury. Tolling events include minority, legal disability, and bankruptcy. Key federal deadlines: 21 days to answer a complaint, 14 days to reply to a counterclaim, 28 days for post-trial motions, 30 days to file a notice of appeal.""",
    },
    {
        "id": "doc_008",
        "topic": "Evidence Rules and Admissibility",
        "text": """The Federal Rules of Evidence govern what evidence may be admitted in federal court. Relevance is the threshold requirement. Even relevant evidence may be excluded if its probative value is substantially outweighed by a danger of unfair prejudice under Rule 403. Hearsay is an out-of-court statement offered to prove the truth of the matter asserted and is generally inadmissible. Common hearsay exceptions include present sense impression, excited utterance, business records, and dying declarations. Admissions by a party-opponent are not hearsay. Character evidence is generally not admissible to prove conforming conduct. Authentication requires sufficient evidence that an item is what it claims to be. Expert testimony is admissible under the Daubert standard if based on sufficient facts, reliable methods reliably applied.""",
    },
    {
        "id": "doc_009",
        "topic": "Bail and Pretrial Detention",
        "text": """Bail is the temporary release of a defendant from custody conditioned on assurance of appearance for future proceedings. The Eighth Amendment prohibits excessive bail. At a bail hearing the court considers: seriousness of the charge, criminal history, community ties, flight risk, and danger to the community. Types of bail include cash bail, surety bond (bondsman posts bail for 10–15% fee), release on recognizance (ROR), and unsecured bond. Conditions of release may include passport surrender, electronic monitoring, curfews, and no-contact orders. Under the Bail Reform Act of 1984, federal courts may order pretrial detention if the government proves by clear and convincing evidence that no release condition would assure appearance and community safety. Detention hearings must be held within 3 business days for flight risk grounds or 5 days for dangerousness.""",
    },
    {
        "id": "doc_010",
        "topic": "Legal Ethics and Attorney-Client Privilege",
        "text": """Attorney-client privilege protects confidential communications between an attorney and client made for the purpose of seeking or providing legal advice. The privilege belongs to the client — only the client can waive it. For the privilege to apply: there must be an attorney-client relationship, the communication must be confidential, it must be for the purpose of legal advice, and the privilege must not have been waived. The crime-fraud exception pierces the privilege when the client sought assistance to engage in a crime or fraud. Waiver can be express or implied. The work product doctrine protects materials prepared by an attorney in anticipation of litigation. Opinion work product receives near-absolute protection. Model Rules of Professional Conduct Rule 1.6 prohibits attorneys from revealing client confidences without informed consent.""",
    },
]

FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES = 2


# ============================================================
# LOAD AGENT (cached — only runs once per session)
# ============================================================
@st.cache_resource
def load_agent():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.Client()
    try:
        client.delete_collection("capstone_kb")
    except:
        pass
    collection = client.create_collection("capstone_kb")

    texts = [d["text"] for d in DOCUMENTS]
    collection.add(
        documents=texts,
        embeddings=embedder.encode(texts).tolist(),
        ids=[d["id"] for d in DOCUMENTS],
        metadatas=[{"topic": d["topic"]} for d in DOCUMENTS],
    )

    # ── Node definitions ───────────────────────────────────
    class CapstoneState(TypedDict):
        question: str
        messages: List[dict]
        route: str
        retrieved: str
        sources: List[str]
        tool_result: str
        answer: str
        faithfulness: float
        eval_retries: int

    def memory_node(state):
        msgs = state.get("messages", [])
        msgs = msgs + [{"role": "user", "content": state["question"]}]
        if len(msgs) > 6:
            msgs = msgs[-6:]
        return {"messages": msgs}

    def router_node(state):
        question = state["question"]
        messages = state.get("messages", [])
        recent = (
            "; ".join(f"{m['role']}: {m['content'][:60]}" for m in messages[-3:-1])
            or "none"
        )
        prompt = f"""You are a router for a Legal Document Assistant used by paralegals and lawyers.
Available routes:
- retrieve: search the legal knowledge base
- memory_only: answer from conversation history (e.g. 'what did you just say?')
- tool: use datetime tool for questions about today's date or filing deadlines

Recent conversation: {recent}
Current question: {question}
Reply with ONLY one word: retrieve / memory_only / tool"""
        decision = llm.invoke(prompt).content.strip().lower()
        if "memory" in decision:
            decision = "memory_only"
        elif "tool" in decision:
            decision = "tool"
        else:
            decision = "retrieve"
        return {"route": decision}

    def retrieval_node(state):
        q_emb = embedder.encode([state["question"]]).tolist()
        results = collection.query(query_embeddings=q_emb, n_results=3)
        chunks = results["documents"][0]
        topics = [m["topic"] for m in results["metadatas"][0]]
        context = "\n\n---\n\n".join(
            f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks))
        )
        return {"retrieved": context, "sources": topics}

    def skip_retrieval_node(state):
        return {"retrieved": "", "sources": []}

    def tool_node(state):
        today = date.today()
        deadlines = {
            "Answer to complaint (federal)": today + timedelta(days=21),
            "Notice of appeal (federal)": today + timedelta(days=30),
            "Post-trial motions (federal)": today + timedelta(days=28),
            "Personal injury SOL (2-year)": today + timedelta(days=730),
            "Breach of contract SOL (4-year)": today + timedelta(days=1460),
        }
        lines = [f"Today's date: {today.strftime('%B %d, %Y')} ({today.isoformat()})"]
        lines.append("\nCommon filing deadlines from today:")
        for label, deadline in deadlines.items():
            lines.append(
                f"  • {label}: {deadline.strftime('%B %d, %Y')} ({(deadline-today).days} days from today)"
            )
        return {"tool_result": "\n".join(lines)}

    def answer_node(state):
        question = state["question"]
        retrieved = state.get("retrieved", "")
        tool_result = state.get("tool_result", "")
        messages = state.get("messages", [])
        eval_retries = state.get("eval_retries", 0)
        context_parts = []
        if retrieved:
            context_parts.append(f"KNOWLEDGE BASE:\n{retrieved}")
        if tool_result:
            context_parts.append(f"TODAY'S DATE & DEADLINES:\n{tool_result}")
        context = "\n\n".join(context_parts)
        if context:
            system_content = f"""You are a Legal Document Assistant helping paralegals and junior lawyers.
Answer using ONLY the information provided in the context below.
If the answer is not in the context, say: I don't have that information in my knowledge base.
Do NOT add information from your training data.

{context}"""
        else:
            system_content = "You are a Legal Document Assistant. Answer based on the conversation history."
        if eval_retries > 0:
            system_content += "\n\nIMPORTANT: Answer using ONLY information explicitly stated in the context above."
        lc_msgs = [SystemMessage(content=system_content)]
        for msg in messages[:-1]:
            lc_msgs.append(
                HumanMessage(content=msg["content"])
                if msg["role"] == "user"
                else AIMessage(content=msg["content"])
            )
        lc_msgs.append(HumanMessage(content=question))
        response = llm.invoke(lc_msgs)
        return {"answer": response.content}

    def eval_node(state):
        answer = state.get("answer", "")
        context = state.get("retrieved", "")[:500]
        retries = state.get("eval_retries", 0)
        if not context:
            return {"faithfulness": 1.0, "eval_retries": retries + 1}
        prompt = f"""Rate faithfulness: does this answer use ONLY information from the context?
Reply with ONLY a number between 0.0 and 1.0.
Context: {context}
Answer: {answer[:300]}"""
        result = llm.invoke(prompt).content.strip()
        try:
            score = float(result.split()[0].replace(",", "."))
            score = max(0.0, min(1.0, score))
        except:
            score = 0.5
        return {"faithfulness": score, "eval_retries": retries + 1}

    def save_node(state):
        messages = state.get("messages", [])
        messages = messages + [{"role": "assistant", "content": state["answer"]}]
        return {"messages": messages}

    def route_decision(state):
        route = state.get("route", "retrieve")
        if route == "tool":
            return "tool"
        if route == "memory_only":
            return "skip"
        return "retrieve"

    def eval_decision(state):
        score = state.get("faithfulness", 1.0)
        retries = state.get("eval_retries", 0)
        if score >= FAITHFULNESS_THRESHOLD or retries >= MAX_EVAL_RETRIES:
            return "save"
        return "answer"

    # ── Assemble graph ─────────────────────────────────────
    g = StateGraph(CapstoneState)
    g.add_node("memory", memory_node)
    g.add_node("router", router_node)
    g.add_node("retrieve", retrieval_node)
    g.add_node("skip", skip_retrieval_node)
    g.add_node("tool", tool_node)
    g.add_node("answer", answer_node)
    g.add_node("eval", eval_node)
    g.add_node("save", save_node)
    g.set_entry_point("memory")
    g.add_edge("memory", "router")
    g.add_conditional_edges(
        "router",
        route_decision,
        {"retrieve": "retrieve", "skip": "skip", "tool": "tool"},
    )
    g.add_edge("retrieve", "answer")
    g.add_edge("skip", "answer")
    g.add_edge("tool", "answer")
    g.add_edge("answer", "eval")
    g.add_conditional_edges("eval", eval_decision, {"answer": "answer", "save": "save"})
    g.add_edge("save", END)

    agent_app = g.compile(checkpointer=MemorySaver())
    return agent_app, embedder, collection


# ── Load everything ────────────────────────────────────────
try:
    agent_app, embedder, collection = load_agent()
    st.success(f"✅ Knowledge base loaded — {collection.count()} documents ready")
except Exception as e:
    st.error(f"Failed to load agent: {e}")
    st.stop()

# ============================================================
# SESSION STATE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]
if "last_meta" not in st.session_state:
    st.session_state.last_meta = {}

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚖️ About")
    st.write("AI-powered legal research assistant for paralegals and junior lawyers.")
    st.divider()

    st.subheader("📚 Topics Covered")
    for d in DOCUMENTS:
        st.write(f"• {d['topic']}")
    st.divider()

    st.subheader("🔖 Session")
    st.code(f"Thread: {st.session_state.thread_id}")

    if st.session_state.last_meta:
        st.subheader("Last Response Info")
        st.write(f"**Route:** {st.session_state.last_meta.get('route', '—')}")
        faith = st.session_state.last_meta.get("faithfulness", 0)
        color = "🟢" if faith >= 0.7 else "🟡"
        st.write(f"**Faithfulness:** {color} {faith:.2f}")
        sources = st.session_state.last_meta.get("sources", [])
        if sources:
            st.write("**Sources:**")
            for s in sources:
                st.write(f"  • {s}")
    st.divider()

    if st.button("🗑️ New Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.session_state.last_meta = {}
        st.rerun()

# ============================================================
# CHAT UI
# ============================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask a legal question..."):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Researching..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = agent_app.invoke({"question": prompt}, config=config)
            answer = result.get("answer", "Sorry, I could not generate an answer.")

        st.write(answer)

        faith = result.get("faithfulness", 0.0)
        sources = result.get("sources", [])
        route = result.get("route", "")

        if sources:
            st.caption(f"📎 Sources: {' · '.join(sources)}")
        if faith > 0:
            color = "🟢" if faith >= 0.7 else "🟡"
            st.caption(f"{color} Faithfulness: {faith:.2f}  |  Route: {route}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.last_meta = {
        "route": route,
        "faithfulness": faith,
        "sources": sources,
    }
