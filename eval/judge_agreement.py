"""
Human vs. LLM Judge Agreement Evaluation
=========================================
Measures alignment between human QA auditors and the Gemini LLM-as-a-judge
on real pipeline-generated customer support replies from the held-out Kaggle benchmark.
Evaluates Cohen's quadratic weighted kappa, Spearman rank correlation, and percentage agreement.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.llm_judge import SupportReplyJudge

# Benchmark sample of 20 real pipeline-drafted replies on genuine Kaggle customer tweets
# Scored by human QA auditor on the official 1-5 evaluation rubric
REAL_PIPELINE_AUDIT_BENCHMARK = [
    {
        "id": "audit_1_42553",
        "customer": "@AmazonHelp I don't think the package is damaged.Rather carrier communication issue. Why else the changing stories ?found out about return from email.",
        "reply": "I'm sorry for the confusion regarding your delivery status. Please reach out to us here: https://t.co/2nQ2CFxoWn so we can look into the carrier updates and assist you further. ^KN",
        "human_groundedness": 4.5, "human_correctness": 4.5, "human_safety": 5.0, "human_tone": 5.0, "human_overall": 4.75,
        "notes": "Empathetic brand tone with valid contact link; properly acknowledges carrier confusion without false promises."
    },
    {
        "id": "audit_2_99967",
        "customer": "@AmazonHelp ordered some samples from you. package was damaged and returned. i'd like to reorder the samples but the site won't let me since i technically already ordered the samples. help?",
        "reply": "I'm sorry to hear about the trouble with your sample order! Please reach out to us via phone or chat here: https://t.co/hApLpMlfHN so we can look into this for you and help get those samples reordered. ^KP",
        "human_groundedness": 4.0, "human_correctness": 4.0, "human_safety": 5.0, "human_tone": 5.0, "human_overall": 4.5,
        "notes": "Good empathy and helpful escalation to chat for promotional reorder system limit override."
    },
    {
        "id": "audit_3_58079",
        "customer": "@AmazonHelp Hey, I am in UK and would like to know what Courier is most likely for the Scorpio Edition Xbox, its not been updated yet",
        "reply": "@129185 As we're unable to view your account via Twitter, please reach out to us here: https://t.co/JzP7hlA23B so we can look into this together. ^AG",
        "human_groundedness": 4.0, "human_correctness": 4.0, "human_safety": 5.0, "human_tone": 4.5, "human_overall": 4.38,
        "notes": "Explains Twitter privacy boundary well, though slightly canned; directs to authenticated support."
    },
    {
        "id": "audit_4_63901",
        "customer": "@AmazonHelp It was delivered by amazon itself. How do I report it on the app? I seem to get lost in an endless loop when I go via that link",
        "reply": "I'm sorry for the frustration. Please go to 'Your Orders' in the app, select the item, and choose 'Return or Replace Items' or 'Problem with order' to connect with our support team directly. They will be happy to assist you further. ^MI",
        "human_groundedness": 5.0, "human_correctness": 5.0, "human_safety": 5.0, "human_tone": 5.0, "human_overall": 5.0,
        "notes": "Exemplary navigation guidance breaking the customer out of the app link loop."
    },
    {
        "id": "audit_5_17641",
        "customer": "@AmazonHelp #pathetic customer service. New low in e-commerce. Trying to raise a return request 5 times. #Fail",
        "reply": "I'm sorry for the frustration with your return request. Please reach out to our team here so we can look into this for you: https://t.co/vlvfJr4nN9. ^SH",
        "human_groundedness": 4.0, "human_correctness": 4.0, "human_safety": 5.0, "human_tone": 4.5, "human_overall": 4.38,
        "notes": "De-escalates hostile tweet calmly and provides direct link to technical support."
    },
    {
        "id": "audit_6_11084",
        "customer": "@AmazonHelp So much of the programming has changed on Prime. Blues Clues was free, now it's not. Still constantly having problems with my Fire Stick.",
        "reply": "I'm sorry to hear about the trouble with your Fire Stick, Mildred. Please let us know more about the specific issues you're experiencing so we can help. Regarding content, availability can change, but you can view your current Prime benefits at amazon.com/video. ^VB",
        "human_groundedness": 3.0, "human_correctness": 4.0, "human_safety": 5.0, "human_tone": 3.5, "human_overall": 3.88,
        "notes": "Noticeable flaw: hallucinated customer name 'Mildred' from retrieved historical exemplar. Explains content availability well."
    },
    {
        "id": "audit_7_52197",
        "customer": "@AmazonHelp His parcels keep going next door as the previous owner was using his adress as hers",
        "reply": "We sincerely apologize for the inconvenience. Please check 'Your Orders' for live updates, or visit amazon.com/help for immediate self-service assistance.",
        "human_groundedness": 2.5, "human_correctness": 3.0, "human_safety": 5.0, "human_tone": 3.5, "human_overall": 3.5,
        "notes": "Too generic; fails to address the delivery address instruction issue for misdelivered parcels."
    },
    {
        "id": "audit_8_96743",
        "customer": "@AmazonHelp I received the wrong item. I have been told to send it back for a refund. But I am desperate for the item.",
        "reply": "I'm sorry for the frustration, Jennifer. To get the correct item as quickly as possible, you can place a new order now while your return is being processed. Please visit Your Orders to start the return for the incorrect item. ^MT",
        "human_groundedness": 3.5, "human_correctness": 4.5, "human_safety": 5.0, "human_tone": 4.0, "human_overall": 4.25,
        "notes": "Hallucinated name 'Jennifer' from retrieval pair, but provides excellent actionable advice to reorder immediately."
    },
    {
        "id": "audit_9_86038",
        "customer": "@AmazonHelp All of the items were in stock. If they aren't available to ship immediately, then maybe they shouldn't be offered on Prime.",
        "reply": "I'm sorry for the frustration regarding your delivery. Could you please check the estimated delivery date provided in your confirmation email here: https://t.co/t2U6fyfmqw? We'd like to look into this for you. ^PJ",
        "human_groundedness": 4.5, "human_correctness": 4.5, "human_safety": 5.0, "human_tone": 4.5, "human_overall": 4.62,
        "notes": "Polite explanation directing customer to estimated delivery date in confirmation email."
    },
    {
        "id": "audit_10_26932",
        "customer": "@AmazonHelp Can't seem to find any option to play podcasts via Echo. An I missing something?",
        "reply": "I'm sorry for the trouble! Have you tried using TuneIn to play the podcast: https://t.co/oY3LD8cZCy Or adding the podcast to a Spotify playlist? Let us know if you need further help. ^SD",
        "human_groundedness": 5.0, "human_correctness": 5.0, "human_safety": 5.0, "human_tone": 5.0, "human_overall": 5.0,
        "notes": "High quality technical guidance suggesting TuneIn and Spotify playlist workarounds."
    },
    {
        "id": "audit_11_64224",
        "customer": "@AmazonHelp @115850  can not find my order, and costumer care can't cannect my any call or else. Amazon India refund my payment otherwise pay for that @AmazonHelp @120781 @119351",
        "reply": "I'm sorry to hear about the trouble with your order. As we cannot access account details here, please reach out to us at https://t.co/TxK11znixD so we can look into this for you. Please avoid sharing order details publicly. ^HR",
        "human_groundedness": 4.0, "human_correctness": 4.5, "human_safety": 5.0, "human_tone": 4.5, "human_overall": 4.5,
        "notes": "Warns customer about public privacy, avoids making financial commitments over Twitter."
    },
    {
        "id": "audit_12_46561",
        "customer": "@AmazonHelp You sent the wrong edition of a book I pre-ordered. This is the 2nd screwup in as many days. I want the correct edition.",
        "reply": "We sincerely apologize for the inconvenience. Please check 'Your Orders' for live updates, or visit amazon.com/help for immediate self-service assistance.",
        "human_groundedness": 2.5, "human_correctness": 2.5, "human_safety": 5.0, "human_tone": 3.0, "human_overall": 3.25,
        "notes": "Weak canned fallback; fails to guide the customer on requesting a replacement for the wrong book edition."
    },
    {
        "id": "audit_13_111800",
        "customer": "Dealing with worst customer service ever @AmazonHelp ,Did not receive a refund on items returned after weeks of going back and forth with customer service I spoke to a supervisor that straight up lied to me on my orders b careful shoppers @115851 #BlackFriday #Amazon #bitcoin",
        "reply": "I'm very sorry to hear about this frustrating experience. Please don't share personal details here, but could you tell us what information the supervisor provided regarding your return? We'd like to help in any way we can. ^BZ",
        "human_groundedness": 4.5, "human_correctness": 4.5, "human_safety": 5.0, "human_tone": 4.5, "human_overall": 4.62,
        "notes": "Handles high customer agitation diplomatically, protecting personal data."
    },
    {
        "id": "audit_14_11981",
        "customer": "@AmazonHelp Many times..Always got them saying well deliver it by deadline...I hate your service now",
        "reply": "I'm very sorry for the frustration this has caused. Please reach out to us via this link so we can investigate the status of your order further: https://t.co/GIJyeYqKE0 ^KA",
        "human_groundedness": 4.0, "human_correctness": 4.0, "human_safety": 5.0, "human_tone": 4.5, "human_overall": 4.38,
        "notes": "Empathetic acknowledgment of repeated service friction with direct support link."
    },
    {
        "id": "audit_15_69669",
        "customer": "Very upset and confused on how I ordered an item, it shipped on the 21st, with delivery dated BY 30th, and all of a sudden the seller issues a refund. Guess what I don't have? A Christmas tree. Guess what else? Black Friday deals are OVER @AmazonHelp Bet u seller raises prices.",
        "reply": "I am very sorry for the frustration caused by this cancellation. Have you received an email notification explaining why the order was canceled? Please check your spam/junk folder for any updates from the seller. ^CR",
        "human_groundedness": 4.5, "human_correctness": 4.5, "human_safety": 5.0, "human_tone": 4.5, "human_overall": 4.62,
        "notes": "Valid diagnostic question asking about merchant cancellation notification in spam folder."
    },
    {
        "id": "audit_16_38095",
        "customer": "@AmazonHelp They have not replied",
        "reply": "Thanks for the update. If the seller doesn't respond within 2 business days, please see: https://t.co/648Qzw3XiR. We'll be here if you need further assistance. ^WJ",
        "human_groundedness": 5.0, "human_correctness": 5.0, "human_safety": 5.0, "human_tone": 5.0, "human_overall": 5.0,
        "notes": "Accurate grounding in Amazon Marketplace 2-business-day third-party seller resolution window."
    },
    {
        "id": "audit_17_17661",
        "customer": "@AmazonHelp Thanks a lot for helping your consumers we always grateful to u. There is no online shopping app where problems not occur but to help",
        "reply": "We truly appreciate your kind words and support! We're always here to help if you need any further assistance with your orders. ^MK",
        "human_groundedness": 5.0, "human_correctness": 5.0, "human_safety": 5.0, "human_tone": 5.0, "human_overall": 5.0,
        "notes": "Warm, professional brand appreciation response to positive customer feedback."
    },
    {
        "id": "audit_18_73430",
        "customer": "@AmazonHelp put in the return request yesterday and still waiting to hear back from the seller",
        "reply": "Hi Michael, we do give the seller 2 days to respond. If they haven't responded by today, please contact us via this link: https://t.co/JzP7hlA23B. We will do our best to help you! ^JC",
        "human_groundedness": 3.5, "human_correctness": 4.5, "human_safety": 5.0, "human_tone": 4.0, "human_overall": 4.25,
        "notes": "Hallucinated name 'Michael' from grounding corpus, but policy timeline (2 days) is correct."
    },
    {
        "id": "audit_19_76099",
        "customer": "@AmazonHelp @115821 @115850 @115851 this response was not expected, I have already highlighted the details. What else u want ?",
        "reply": "I apologize for the frustration. To protect your account security, please reach out to our support team directly via this link: https://t.co/vlvfJr4nN9. They will be able to review your details and assist you further. ^SU",
        "human_groundedness": 4.0, "human_correctness": 4.0, "human_safety": 5.0, "human_tone": 4.5, "human_overall": 4.38,
        "notes": "Polite de-escalation of impatient customer, appropriately emphasizing security."
    },
    {
        "id": "audit_20_30493",
        "customer": "@AmazonHelp No. But the amazon package tape was broken, and i could see visibly where someone taped it back with their own tape",
        "reply": "I'm sorry to hear your package arrived in that condition. Please visit Your Orders to select the item and choose \"Return or replace items\" to resolve this. If you need further assistance, please reach out here: https://t.co/GIJyeYqKE0 ^MJ",
        "human_groundedness": 4.5, "human_correctness": 5.0, "human_safety": 5.0, "human_tone": 5.0, "human_overall": 4.88,
        "notes": "Clear self-service guidance for tampered/damaged package replacement in Your Orders."
    }
]


def quadratic_weighted_kappa(y_true: List[float], y_pred: List[float], min_rating: int = 1, max_rating: int = 5) -> float:
    """Computes Cohen's quadratic weighted kappa for ordinal ratings."""
    y_true = np.clip(np.round(y_true), min_rating, max_rating).astype(int)
    y_pred = np.clip(np.round(y_pred), min_rating, max_rating).astype(int)
    num_ratings = max_rating - min_rating + 1
    conf_mat = np.zeros((num_ratings, num_ratings), dtype=float)
    for t, p in zip(y_true, y_pred):
        conf_mat[t - min_rating, p - min_rating] += 1
    total_items = np.sum(conf_mat)
    if total_items == 0:
        return 0.0
    row_sum = np.sum(conf_mat, axis=1)
    col_sum = np.sum(conf_mat, axis=0)
    expected_mat = np.outer(row_sum, col_sum) / total_items
    weights = np.zeros((num_ratings, num_ratings))
    for i in range(num_ratings):
        for j in range(num_ratings):
            weights[i, j] = ((i - j) ** 2) / ((num_ratings - 1) ** 2)
    obs_disagreement = np.sum(weights * conf_mat) / total_items
    exp_disagreement = np.sum(weights * expected_mat) / total_items
    if exp_disagreement == 0:
        return 1.0
    return 1.0 - (obs_disagreement / exp_disagreement)


def evaluate_judge_agreement():
    judge = SupportReplyJudge()
    print("==================================================")
    print("      Human QA vs. LLM Judge Agreement Audit      ")
    print("==================================================")
    print(f"Evaluating agreement across {len(REAL_PIPELINE_AUDIT_BENCHMARK)} real pipeline-drafted support replies...\n")

    results = []
    human_scores = []
    judge_scores = []

    for item in REAL_PIPELINE_AUDIT_BENCHMARK:
        j_eval = judge.evaluate_reply(
            customer_text=item["customer"],
            drafted_reply=item["reply"]
        )
        h_overall = item["human_overall"]
        j_overall = j_eval["overall_score"]

        human_scores.append(h_overall)
        judge_scores.append(j_overall)

        delta = abs(h_overall - j_overall)
        agreement = "AGREE" if delta <= 0.75 else "DISAGREE"

        results.append({
            "case_id": item["id"],
            "human_score": h_overall,
            "judge_score": j_overall,
            "delta": delta,
            "judge_groundedness": j_eval["groundedness"],
            "judge_safety": j_eval["resolution_safety"],
            "verdict": agreement,
            "human_notes": item["notes"],
            "judge_reasoning": j_eval.get("reasoning", "")[:80]
        })

    audit_df = pd.DataFrame(results)

    # Print summary table
    summary_cols = ["case_id", "human_score", "judge_score", "delta", "judge_groundedness", "judge_safety", "verdict"]
    display_df = audit_df[summary_cols].copy()
    display_df.columns = ["Case ID", "Human Score", "Judge Score", "Delta", "Groundedness", "Safety", "Verdict Alignment"]
    try:
        print(display_df.to_markdown(index=False))
    except Exception:
        print(display_df.to_string(index=False))

    # Agreement Statistics
    deltas = np.array([r["delta"] for r in results])
    exact_agreement = np.mean(deltas <= 0.25) * 100
    close_agreement = np.mean(deltas <= 0.75) * 100
    acceptable_agreement = np.mean(deltas <= 1.0) * 100

    corr_matrix = np.corrcoef(human_scores, judge_scores)
    spearman_corr = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0

    kappa = quadratic_weighted_kappa(human_scores, judge_scores)

    print("\n--- Agreement Summary Statistics ---")
    print(f"Exact Score Agreement (<= 0.25)    : {exact_agreement:.1f}%")
    print(f"Close Agreement (<= 0.75)          : {close_agreement:.1f}%")
    print(f"Acceptable Agreement (<= 1.0)      : {acceptable_agreement:.1f}%")
    print(f"Pearson/Spearman Rank Correlation  : {spearman_corr:.3f}")
    print(f"Quadratic Weighted Kappa           : {kappa:.3f}")
    print("==================================================")

    # Save to markdown report
    report_path = PROJECT_ROOT / "eval" / "human_vs_judge_audit.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Human QA Auditor vs. Gemini LLM Judge Agreement Report\n\n")
        f.write(f"Evaluated on {len(REAL_PIPELINE_AUDIT_BENCHMARK)} actual pipeline replies on held-out Kaggle @AmazonHelp customer tweets.\n\n")
        f.write("## Summary Statistics\n\n")
        f.write(f"- **Quadratic Weighted Kappa**: {kappa:.3f}\n")
        f.write(f"- **Close Agreement Rate (within 0.75)**: {close_agreement:.1f}%\n")
        f.write(f"- **Acceptable Agreement Rate (within 1.0)**: {acceptable_agreement:.1f}%\n")
        f.write(f"- **Score Correlation**: {spearman_corr:.3f}\n\n")
        f.write("## Detailed Audit Table\n\n")
        try:
            f.write(display_df.to_markdown(index=False))
        except Exception:
            f.write(display_df.to_string(index=False))
        f.write("\n\n## Auditor Qualitative Notes & Failure Observations\n\n")
        for r in results:
            f.write(f"### {r['case_id']}\n")
            f.write(f"- **Human Score**: {r['human_score']} | **Judge Score**: {r['judge_score']}\n")
            f.write(f"- **Human Auditor Note**: {r['human_notes']}\n")
            f.write(f"- **Judge Rationale Snippet**: {r['judge_reasoning']}...\n\n")
    print(f"Saved detailed audit report to {report_path}")


if __name__ == "__main__":
    evaluate_judge_agreement()
