"""LLM-powered compliance alert generation with template fallback."""

import os
from dotenv import load_dotenv

load_dotenv()


def get_risk_level(anomaly_score: float) -> str:
    if anomaly_score >= 0.8:
        return "CRITICAL"
    elif anomaly_score >= 0.6:
        return "HIGH"
    elif anomaly_score >= 0.3:
        return "MEDIUM"
    return "LOW"


def generate_risk_summary(user_id, anomaly_score, top_features, risk_level):
    """Generate compliance summary using Anthropic, fallback to template on failure."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()

    if not api_key:
        return _fallback_alert(user_id, anomaly_score, top_features, risk_level)

    try:
        import anthropic

        feature_lines = "\n".join(
            [f"- {f['feature']}: {f['value']}" for f in top_features]
        )
        if not feature_lines:
            feature_lines = "- No top feature data available"

        prompt = f"""You are a compliance analyst at a forex brokerage.
Write one concise alert paragraph (3-4 sentences) for a suspicious user.

User ID: {user_id}
Anomaly Score: {anomaly_score:.4f}
Risk Level: {risk_level}
Top Contributing Features:
{feature_lines}

Requirements:
- Professional and actionable tone
- Mention concrete risk indicators from the features
- Suggest immediate next action for compliance
- Keep it short and clear"""

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=220,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )

        if message.content and hasattr(message.content[0], "text"):
            return message.content[0].text.strip()
    except Exception:
        pass

    return _fallback_alert(user_id, anomaly_score, top_features, risk_level)


def _fallback_alert(user_id, anomaly_score, top_features, risk_level):
    if top_features:
        primary = top_features[0]
        desc = _describe_feature(primary["feature"], primary["value"])
    else:
        desc = "unusual behavioral patterns"

    secondary = ""
    if len(top_features) > 1:
        names = [f["feature"].replace("_", " ") for f in top_features[1:]]
        secondary = f" Additional risk signals: elevated {' and '.join(names)}."

    return (
        f"COMPLIANCE ALERT -- User {user_id} flagged as {risk_level} risk "
        f"(score: {anomaly_score:.2f}). "
        f"Primary concern: {desc}.{secondary} "
        f"Recommend compliance review and potential account freeze."
    )


def _describe_feature(name, value):
    """Turn a feature name + value into something a human can read."""
    descriptions = {
        "structuring_score": f"multiple deposits under reporting threshold ({int(value)} flagged)",
        "night_login_ratio": f"high proportion of late-night logins ({value:.0%})",
        "unique_countries_7d": f"logins from {int(value)} countries in 7 days",
        "unique_ips_24h": f"access from {int(value)} IPs in 24 hours",
        "deposit_withdrawal_ratio": f"suspicious deposit/withdrawal pattern (ratio: {value:.2f})",
        "trade_volume_zscore": f"extreme trade volume deviation (z-score: {value:.2f})",
        "max_volume_spike_ratio": f"volume spike of {value:.1f}x normal",
        "ip_switch_rate": f"rapid IP switching (rate: {value:.2f})",
        "rapid_navigation_score": f"bot-like navigation (score: {value:.2f})",
        "total_deposits_7d": f"high deposit activity (${value:,.2f} in 7 days)",
        "total_withdrawals_7d": f"large withdrawals (${value:,.2f} in 7 days)",
        "consistent_profit_score": f"suspiciously consistent profits ({value:.0%} win rate)",
        "login_count_24h": f"excessive logins ({int(value)} in 24 hours)",
    }
    return descriptions.get(name, f"unusual {name.replace('_', ' ')} ({value:.2f})")
