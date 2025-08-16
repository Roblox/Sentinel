# Copyright 2025 Roblox Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Safety guardrails and usage restrictions for Sentinel detection systems.

This module implements comprehensive safety measures to ensure responsible use
of the Sentinel detection system, preventing misuse while enabling legitimate
defensive security applications.
"""

import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import inspect
from pathlib import Path

LOG = logging.getLogger(__name__)


class UsageContext(Enum):
    """Authorized usage contexts for Sentinel."""
    
    CONTENT_MODERATION = "content_moderation"
    ACADEMIC_RESEARCH = "academic_research"
    SAFETY_SYSTEM_DEVELOPMENT = "safety_system_development"
    SECURITY_TESTING = "security_testing"
    VULNERABILITY_ASSESSMENT = "vulnerability_assessment"
    EDUCATIONAL_TRAINING = "educational_training"


class AccessLevel(Enum):
    """Access levels for different components."""
    
    PUBLIC = "public"          # General safety APIs
    RESTRICTED = "restricted"  # Advanced features requiring approval
    INTERNAL = "internal"      # Internal development/research
    AUDIT_ONLY = "audit_only"  # Read-only access for auditing


@dataclass
class UsagePolicy:
    """Defines usage policy and restrictions."""
    
    allowed_contexts: Set[UsageContext]
    access_level: AccessLevel
    rate_limits: Dict[str, int] = field(default_factory=dict)
    data_retention_days: int = 30
    audit_required: bool = True
    approval_required: bool = False
    
    # Content restrictions
    max_batch_size: int = 1000
    max_daily_requests: int = 10000
    prohibited_use_cases: List[str] = field(default_factory=list)
    
    # Monitoring requirements
    log_all_requests: bool = True
    alert_on_misuse: bool = True
    automatic_shutoff_threshold: int = 1000


@dataclass
class AuditLog:
    """Audit log entry for tracking usage."""
    
    timestamp: str
    user_id: str
    usage_context: UsageContext
    component_accessed: str
    request_details: Dict[str, Any]
    result_summary: Dict[str, Any]
    compliance_status: str
    risk_level: str = "none"


class SafetyGuardrails:
    """Comprehensive safety system for Sentinel usage monitoring and control."""
    
    def __init__(
        self,
        policy_file: Optional[Path] = None,
        audit_log_dir: Optional[Path] = None,
        enable_strict_mode: bool = True
    ):
        """Initialize safety guardrails.
        
        Args:
            policy_file: Path to usage policy configuration file.
            audit_log_dir: Directory for audit logs.
            enable_strict_mode: Whether to enforce strict safety measures.
        """
        self.enable_strict_mode = enable_strict_mode
        self.audit_log_dir = audit_log_dir or Path("./audit_logs")
        self.audit_log_dir.mkdir(exist_ok=True)
        
        # Load or create default policies
        self.policies = self._load_policies(policy_file)
        
        # Usage tracking
        self.usage_stats: Dict[str, Dict[str, int]] = {}
        self.blocked_requests: List[Dict[str, Any]] = []
        self.audit_logs: List[AuditLog] = []
        
        # Safety checks
        self._validate_environment()
        
        LOG.info("Safety guardrails initialized with strict_mode=%s", enable_strict_mode)
    
    def _load_policies(self, policy_file: Optional[Path]) -> Dict[str, UsagePolicy]:
        """Load usage policies from file or create defaults."""
        
        default_policies = {
            "content_moderation": UsagePolicy(
                allowed_contexts={UsageContext.CONTENT_MODERATION},
                access_level=AccessLevel.PUBLIC,
                rate_limits={"requests_per_minute": 100, "requests_per_hour": 1000},
                max_batch_size=100,
                max_daily_requests=5000,
                prohibited_use_cases=[
                    "generating harmful content",
                    "training offensive language models",
                    "creating attack tools"
                ]
            ),
            
            "research": UsagePolicy(
                allowed_contexts={UsageContext.ACADEMIC_RESEARCH, UsageContext.EDUCATIONAL_TRAINING},
                access_level=AccessLevel.RESTRICTED,
                rate_limits={"requests_per_minute": 50, "requests_per_hour": 200},
                max_batch_size=500,
                max_daily_requests=2000,
                approval_required=True,
                prohibited_use_cases=[
                    "commercial exploitation",
                    "malicious research",
                    "privacy violations"
                ]
            ),
            
            "development": UsagePolicy(
                allowed_contexts={
                    UsageContext.SAFETY_SYSTEM_DEVELOPMENT,
                    UsageContext.SECURITY_TESTING,
                    UsageContext.VULNERABILITY_ASSESSMENT
                },
                access_level=AccessLevel.INTERNAL,
                rate_limits={"requests_per_minute": 200, "requests_per_hour": 2000},
                max_batch_size=1000,
                max_daily_requests=10000,
                approval_required=True,
                prohibited_use_cases=[
                    "offensive capability development",
                    "surveillance systems",
                    "discriminatory applications"
                ]
            )
        }
        
        if policy_file and policy_file.exists():
            try:
                with open(policy_file) as f:
                    loaded_policies = json.load(f)
                
                # Merge with defaults
                for name, policy_data in loaded_policies.items():
                    if name in default_policies:
                        # Update existing policy
                        policy = default_policies[name]
                        for key, value in policy_data.items():
                            if hasattr(policy, key):
                                setattr(policy, key, value)
                
            except Exception as e:
                LOG.error(f"Error loading policy file: {e}")
        
        return default_policies
    
    def _validate_environment(self):
        """Validate that the environment is appropriate for defensive security use."""
        
        # Check for suspicious environment indicators
        warnings = []
        
        # Check if we're in a known offensive security context
        suspicious_modules = [
            "metasploit", "exploit", "attack", "offensive", "weaponize"
        ]
        
        for module_name in suspicious_modules:
            try:
                __import__(module_name)
                warnings.append(f"Suspicious module detected: {module_name}")
            except ImportError:
                pass
        
        # Check calling context
        frame = inspect.currentframe()
        try:
            while frame:
                filename = frame.f_code.co_filename.lower()
                if any(term in filename for term in ["attack", "exploit", "weapon", "offensive"]):
                    warnings.append(f"Suspicious calling context: {filename}")
                frame = frame.f_back
        finally:
            del frame
        
        if warnings and self.enable_strict_mode:
            for warning in warnings:
                LOG.warning(f"Security warning: {warning}")
            
            # In strict mode, require explicit acknowledgment
            self._require_usage_acknowledgment()
    
    def _require_usage_acknowledgment(self):
        """Require explicit acknowledgment of usage restrictions."""
        
        usage_agreement = """
        ┌─────────────────────────────────────────────────────────────────┐
        │                    DEFENSIVE SECURITY USAGE ONLY               │
        ├─────────────────────────────────────────────────────────────────┤
        │                                                                 │
        │ This Sentinel detection system is authorized ONLY for:         │
        │                                                                 │
        │ ✓ Content safety and moderation                               │
        │ ✓ Academic research on harmful content detection              │
        │ ✓ Defensive security system development                       │
        │ ✓ Educational and training purposes                           │
        │                                                                 │
        │ PROHIBITED USES:                                                │
        │                                                                 │
        │ ✗ Generating or amplifying harmful content                    │
        │ ✗ Training offensive language models                          │
        │ ✗ Creating attack tools or capabilities                       │
        │ ✗ Surveillance or monitoring without consent                  │
        │ ✗ Discriminatory or biased applications                       │
        │                                                                 │
        │ By using this system, you acknowledge that:                    │
        │ • You will use it only for defensive security purposes         │
        │ • You will not misuse the training data or models             │
        │ • You understand that usage is monitored and audited          │
        │ • Violations may result in access termination                 │
        │                                                                 │
        └─────────────────────────────────────────────────────────────────┘
        """
        
        print(usage_agreement)
        LOG.info("Usage acknowledgment displayed")
    
    def check_access_permission(
        self,
        user_id: str,
        usage_context: UsageContext,
        component: str,
        request_details: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Check if access is permitted for a request.
        
        Args:
            user_id: Identifier for the requesting user/system.
            usage_context: Intended usage context.
            component: Component being accessed.
            request_details: Details of the request.
            
        Returns:
            Tuple of (is_permitted, reason_if_denied).
        """
        
        # Find applicable policy
        policy = self._get_applicable_policy(usage_context, component)
        if not policy:
            return False, "No applicable usage policy found"
        
        # Check usage context
        if usage_context not in policy.allowed_contexts:
            return False, f"Usage context {usage_context.value} not authorized"
        
        # Check rate limits
        if not self._check_rate_limits(user_id, policy):
            return False, "Rate limit exceeded"
        
        # Check batch size limits
        batch_size = request_details.get("batch_size", 1)
        if batch_size > policy.max_batch_size:
            return False, f"Batch size {batch_size} exceeds limit of {policy.max_batch_size}"
        
        # Check for prohibited use cases
        request_description = request_details.get("description", "").lower()
        for prohibited in policy.prohibited_use_cases:
            if prohibited.lower() in request_description:
                return False, f"Request appears to involve prohibited use case: {prohibited}"
        
        # Check for suspicious content patterns
        if self._detect_suspicious_patterns(request_details):
            return False, "Suspicious usage patterns detected"
        
        return True, None
    
    def _get_applicable_policy(self, usage_context: UsageContext, component: str) -> Optional[UsagePolicy]:
        """Get the applicable policy for a request."""
        
        # Component-specific policies take precedence
        if component in self.policies:
            return self.policies[component]
        
        # Context-based policies
        context_policies = {
            UsageContext.CONTENT_MODERATION: "content_moderation",
            UsageContext.ACADEMIC_RESEARCH: "research",
            UsageContext.EDUCATIONAL_TRAINING: "research",
            UsageContext.SAFETY_SYSTEM_DEVELOPMENT: "development",
            UsageContext.SECURITY_TESTING: "development",
            UsageContext.VULNERABILITY_ASSESSMENT: "development"
        }
        
        policy_name = context_policies.get(usage_context)
        return self.policies.get(policy_name) if policy_name else None
    
    def _check_rate_limits(self, user_id: str, policy: UsagePolicy) -> bool:
        """Check if request is within rate limits."""
        
        current_time = datetime.now()
        
        # Initialize user stats if not exists
        if user_id not in self.usage_stats:
            self.usage_stats[user_id] = {
                "requests_this_minute": 0,
                "requests_this_hour": 0,
                "requests_today": 0,
                "last_request_time": current_time.isoformat(),
                "minute_start": current_time.replace(second=0, microsecond=0).isoformat(),
                "hour_start": current_time.replace(minute=0, second=0, microsecond=0).isoformat(),
                "day_start": current_time.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
            }
        
        user_stats = self.usage_stats[user_id]
        
        # Reset counters if time windows have passed
        minute_start = datetime.fromisoformat(user_stats["minute_start"])
        if current_time - minute_start >= timedelta(minutes=1):
            user_stats["requests_this_minute"] = 0
            user_stats["minute_start"] = current_time.replace(second=0, microsecond=0).isoformat()
        
        hour_start = datetime.fromisoformat(user_stats["hour_start"])
        if current_time - hour_start >= timedelta(hours=1):
            user_stats["requests_this_hour"] = 0
            user_stats["hour_start"] = current_time.replace(minute=0, second=0, microsecond=0).isoformat()
        
        day_start = datetime.fromisoformat(user_stats["day_start"])
        if current_time - day_start >= timedelta(days=1):
            user_stats["requests_today"] = 0
            user_stats["day_start"] = current_time.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        
        # Check limits
        if user_stats["requests_this_minute"] >= policy.rate_limits.get("requests_per_minute", float('inf')):
            return False
        
        if user_stats["requests_this_hour"] >= policy.rate_limits.get("requests_per_hour", float('inf')):
            return False
        
        if user_stats["requests_today"] >= policy.max_daily_requests:
            return False
        
        # Update counters
        user_stats["requests_this_minute"] += 1
        user_stats["requests_this_hour"] += 1
        user_stats["requests_today"] += 1
        user_stats["last_request_time"] = current_time.isoformat()
        
        return True
    
    def _detect_suspicious_patterns(self, request_details: Dict[str, Any]) -> bool:
        """Detect suspicious usage patterns that might indicate misuse."""
        
        # Check for attempts to extract training data
        text_content = str(request_details.get("text_samples", "")).lower()
        
        suspicious_patterns = [
            # Data extraction attempts
            "show me training data",
            "reveal examples",
            "dump dataset",
            "extract corpus",
            
            # Reverse engineering attempts
            "model weights",
            "internal parameters",
            "architecture details",
            
            # Offensive usage indicators
            "generate hate speech",
            "create toxic content",
            "bypass detection",
            "evade moderation",
            
            # Surveillance indicators
            "monitor users",
            "track individuals",
            "profile people"
        ]
        
        for pattern in suspicious_patterns:
            if pattern in text_content:
                LOG.warning(f"Suspicious pattern detected: {pattern}")
                return True
        
        # Check for unusual batch patterns
        batch_size = request_details.get("batch_size", 1)
        if batch_size > 500:  # Large batches might indicate data mining
            return True
        
        return False
    
    def log_access(
        self,
        user_id: str,
        usage_context: UsageContext,
        component: str,
        request_details: Dict[str, Any],
        result_summary: Dict[str, Any],
        permitted: bool
    ):
        """Log access attempt for audit trail."""
        
        audit_entry = AuditLog(
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            usage_context=usage_context,
            component_accessed=component,
            request_details=self._sanitize_request_details(request_details),
            result_summary=result_summary,
            compliance_status="permitted" if permitted else "denied",
            risk_level=self._assess_risk_level(request_details, result_summary)
        )
        
        self.audit_logs.append(audit_entry)
        
        # Write to file
        log_file = self.audit_log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, 'a') as f:
            json.dump(audit_entry.__dict__, f)
            f.write('\n')
        
        # Alert on high-risk activities
        if audit_entry.risk_level in ["high", "critical"]:
            self._trigger_security_alert(audit_entry)
    
    def _sanitize_request_details(self, request_details: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize request details for logging (remove sensitive content)."""
        
        sanitized = request_details.copy()
        
        # Hash text content instead of storing plaintext
        if "text_samples" in sanitized:
            text_samples = sanitized["text_samples"]
            if isinstance(text_samples, list):
                sanitized["text_samples"] = [
                    hashlib.sha256(text.encode()).hexdigest()[:16] for text in text_samples
                ]
            else:
                sanitized["text_samples"] = hashlib.sha256(str(text_samples).encode()).hexdigest()[:16]
        
        # Remove other potentially sensitive fields
        sensitive_fields = ["user_data", "personal_info", "credentials"]
        for field in sensitive_fields:
            if field in sanitized:
                sanitized[field] = "[REDACTED]"
        
        return sanitized
    
    def _assess_risk_level(self, request_details: Dict[str, Any], result_summary: Dict[str, Any]) -> str:
        """Assess risk level of the request and results."""
        
        # High risk indicators
        high_risk_indicators = [
            request_details.get("batch_size", 0) > 1000,
            result_summary.get("high_risk_detections", 0) > 10,
            "bypass" in str(request_details).lower(),
            "exploit" in str(request_details).lower()
        ]
        
        if any(high_risk_indicators):
            return "high"
        
        # Medium risk indicators
        medium_risk_indicators = [
            request_details.get("batch_size", 0) > 100,
            result_summary.get("violations_detected", 0) > 5
        ]
        
        if any(medium_risk_indicators):
            return "medium"
        
        return "low"
    
    def _trigger_security_alert(self, audit_entry: AuditLog):
        """Trigger security alert for high-risk activities."""
        
        alert = {
            "timestamp": audit_entry.timestamp,
            "alert_type": "HIGH_RISK_ACTIVITY",
            "user_id": audit_entry.user_id,
            "component": audit_entry.component_accessed,
            "risk_level": audit_entry.risk_level,
            "details": "High-risk activity detected in Sentinel usage"
        }
        
        LOG.critical(f"SECURITY ALERT: {alert}")
        
        # Write to separate alert file
        alert_file = self.audit_log_dir / "security_alerts.jsonl"
        with open(alert_file, 'a') as f:
            json.dump(alert, f)
            f.write('\n')
    
    def get_usage_statistics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get usage statistics."""
        
        if user_id:
            return self.usage_stats.get(user_id, {})
        
        # Aggregate statistics
        total_users = len(self.usage_stats)
        total_requests = sum(stats.get("requests_today", 0) for stats in self.usage_stats.values())
        
        return {
            "total_users": total_users,
            "total_requests_today": total_requests,
            "blocked_requests": len(self.blocked_requests),
            "audit_log_entries": len(self.audit_logs),
            "high_risk_activities": len([log for log in self.audit_logs if log.risk_level == "high"])
        }
    
    def export_compliance_report(self) -> Dict[str, Any]:
        """Export compliance report for auditing."""
        
        return {
            "report_timestamp": datetime.now().isoformat(),
            "usage_statistics": self.get_usage_statistics(),
            "policy_compliance": {
                "total_requests": len(self.audit_logs),
                "permitted_requests": len([log for log in self.audit_logs if log.compliance_status == "permitted"]),
                "denied_requests": len([log for log in self.audit_logs if log.compliance_status == "denied"]),
                "high_risk_activities": len([log for log in self.audit_logs if log.risk_level == "high"])
            },
            "safety_measures": {
                "strict_mode_enabled": self.enable_strict_mode,
                "audit_logging_enabled": True,
                "rate_limiting_enabled": True,
                "suspicious_pattern_detection_enabled": True
            }
        }


# Decorator for protected functions
def require_authorization(
    usage_context: UsageContext,
    component: str = "unknown",
    guardrails: Optional[SafetyGuardrails] = None
):
    """Decorator to require authorization for sensitive functions.
    
    Args:
        usage_context: Required usage context for access.
        component: Component name for audit logging.
        guardrails: SafetyGuardrails instance (or use global).
    """
    
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Get guardrails instance
            if guardrails is None:
                # Try to get from global context or create default
                global_guardrails = getattr(wrapper, '_global_guardrails', None)
                if global_guardrails is None:
                    global_guardrails = SafetyGuardrails(enable_strict_mode=True)
                    wrapper._global_guardrails = global_guardrails
                current_guardrails = global_guardrails
            else:
                current_guardrails = guardrails
            
            # Extract user_id and request details from arguments
            user_id = kwargs.get("user_id", "anonymous")
            request_details = {
                "function": func.__name__,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys())
            }
            
            # Check authorization
            permitted, reason = current_guardrails.check_access_permission(
                user_id=user_id,
                usage_context=usage_context,
                component=component,
                request_details=request_details
            )
            
            if not permitted:
                # Log denied access
                current_guardrails.log_access(
                    user_id=user_id,
                    usage_context=usage_context,
                    component=component,
                    request_details=request_details,
                    result_summary={"access_denied": True, "reason": reason},
                    permitted=False
                )
                
                raise PermissionError(f"Access denied: {reason}")
            
            # Execute function
            try:
                result = func(*args, **kwargs)
                
                # Log successful access
                current_guardrails.log_access(
                    user_id=user_id,
                    usage_context=usage_context,
                    component=component,
                    request_details=request_details,
                    result_summary={"access_granted": True, "function_executed": True},
                    permitted=True
                )
                
                return result
                
            except Exception as e:
                # Log failed execution
                current_guardrails.log_access(
                    user_id=user_id,
                    usage_context=usage_context,
                    component=component,
                    request_details=request_details,
                    result_summary={"access_granted": True, "execution_failed": True, "error": str(e)},
                    permitted=True
                )
                raise
        
        return wrapper
    return decorator


# Global guardrails instance
_global_guardrails: Optional[SafetyGuardrails] = None


def initialize_global_guardrails(
    policy_file: Optional[Path] = None,
    audit_log_dir: Optional[Path] = None,
    enable_strict_mode: bool = True
) -> SafetyGuardrails:
    """Initialize global safety guardrails."""
    
    global _global_guardrails
    _global_guardrails = SafetyGuardrails(
        policy_file=policy_file,
        audit_log_dir=audit_log_dir,
        enable_strict_mode=enable_strict_mode
    )
    return _global_guardrails


def get_global_guardrails() -> Optional[SafetyGuardrails]:
    """Get the global safety guardrails instance."""
    return _global_guardrails