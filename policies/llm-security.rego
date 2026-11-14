# LLM Security Policies (OPA/Rego)
#
# Enforces OWASP LLM Top 10 controls and EU AI Act requirements
# for all AI/LLM interactions.

package sentinel.llm_security

import future.keywords.in
import future.keywords.if

# -------------------------------------------------------------------
# LLM01 — Prompt Injection Prevention
# -------------------------------------------------------------------

deny_prompt_injection[msg] if {
    input.type == "llm_request"
    pattern := injection_patterns[_]
    regex.match(pattern, lower(input.user_message))
    msg := sprintf("Prompt injection detected: %s", [pattern])
}

injection_patterns := [
    "ignore\\s+(all\\s+)?previous\\s+(instructions|prompts)",
    "disregard\\s+(all\\s+)?previous",
    "you\\s+are\\s+now\\s+(a|an)\\s+",
    "system\\s*:\\s*you\\s+(are|must|should)",
    "\\[system\\]",
    "\\[inst\\]",
    "repeat.*system.*prompt",
    "show.*your.*(system|hidden).*prompt",
    "jailbreak",
    "bypass\\s+(all\\s+)?safety",
    "do\\s+anything\\s+now",
]

# -------------------------------------------------------------------
# LLM02 — Sensitive Data Protection
# -------------------------------------------------------------------

deny_unredacted_pii[msg] if {
    input.type == "llm_request"
    not input.pii_redacted
    pii_type := pii_patterns[_]
    regex.match(pii_type.pattern, input.user_message)
    msg := sprintf("Unredacted %s in LLM request", [pii_type.name])
}

pii_patterns := [
    {"name": "SSN", "pattern": "\\d{3}-\\d{2}-\\d{4}"},
    {"name": "credit_card", "pattern": "\\d{4}[-\\s]?\\d{4}[-\\s]?\\d{4}[-\\s]?\\d{4}"},
    {"name": "AWS_access_key", "pattern": "AKIA[0-9A-Z]{16}"},
]

# -------------------------------------------------------------------
# LLM03 — Tool Permission Enforcement
# -------------------------------------------------------------------

deny_tool_unauthorized[msg] if {
    input.type == "tool_execution"
    tool := tool_permissions[input.tool_name]
    not input.user_role in tool.allowed_roles
    msg := sprintf("Role '%s' not authorized for tool '%s'", [input.user_role, input.tool_name])
}

require_approval[msg] if {
    input.type == "tool_execution"
    tool := tool_permissions[input.tool_name]
    tool.requires_approval
    not input.approval_granted
    msg := sprintf("Tool '%s' requires human approval", [input.tool_name])
}

tool_permissions := {
    "read_logs": {"allowed_roles": ["admin", "analyst", "operator"], "requires_approval": false, "risk_level": "low"},
    "scan_threats": {"allowed_roles": ["admin", "analyst"], "requires_approval": false, "risk_level": "low"},
    "block_ip": {"allowed_roles": ["admin", "operator"], "requires_approval": true, "risk_level": "high"},
    "kill_container": {"allowed_roles": ["admin"], "requires_approval": true, "risk_level": "critical"},
    "rotate_credentials": {"allowed_roles": ["admin"], "requires_approval": true, "risk_level": "critical"},
}

# -------------------------------------------------------------------
# LLM04 — Rate Limits & Budget Controls
# -------------------------------------------------------------------

deny_budget_exceeded[msg] if {
    input.type == "llm_request"
    input.tenant_tokens_used_today + input.estimated_tokens > input.tenant_daily_limit
    msg := sprintf("Daily token budget exceeded: %d + %d > %d", [
        input.tenant_tokens_used_today, input.estimated_tokens, input.tenant_daily_limit,
    ])
}

# -------------------------------------------------------------------
# LLM05 — Output Validation
# -------------------------------------------------------------------

flag_system_prompt_leak[msg] if {
    input.type == "llm_response"
    pattern := system_leak_patterns[_]
    regex.match(pattern, lower(input.llm_output))
    msg := "Potential system prompt leak in LLM output"
}

system_leak_patterns := [
    "my\\s+system\\s+prompt\\s+is",
    "my\\s+instructions\\s+are",
    "i\\s+was\\s+told\\s+to\\s+(never|always|ignore)",
]

# -------------------------------------------------------------------
# EU AI Act Compliance
# -------------------------------------------------------------------

require_transparency_notice[msg] if {
    input.type == "ai_interaction"
    ai_system := ai_risk_classification[input.ai_system]
    ai_system.risk_level == "high"
    not input.transparency_notice_presented
    msg := sprintf("High-risk AI system '%s' requires transparency notice (EU AI Act Art. 52)", [input.ai_system])
}

ai_risk_classification := {
    "threat_detection": {"risk_level": "high", "article_14_required": true},
    "security_copilot": {"risk_level": "high", "article_14_required": true},
    "compliance_ai": {"risk_level": "high", "article_14_required": true},
    "anomaly_detection": {"risk_level": "limited", "article_14_required": false},
    "vulnerability_scorer": {"risk_level": "limited", "article_14_required": false},
}
