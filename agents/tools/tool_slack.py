"""
Slack Tool Wrapper.

Provides functions to send messages and reports to Slack.
Mockable for testing.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Standard result from tool execution."""
    success: bool
    data: Any
    error: Optional[str] = None
    requires_confirmation: bool = False
    action_description: Optional[str] = None


class SlackTool:
    """
    Tool wrapper for Slack operations.
    
    Supports both webhook and Bot Token API methods.
    """
    
    def __init__(
        self,
        webhook_url: Optional[str] = None,
        bot_token: Optional[str] = None,
        default_channel: str = "#alerts",
        timeout: int = 10,
    ):
        """
        Initialize Slack tool.
        
        Args:
            webhook_url: Incoming webhook URL (simplest method).
            bot_token: Bot OAuth token (for full API access).
            default_channel: Default channel for messages.
            timeout: Request timeout in seconds.
        """
        self.webhook_url = webhook_url
        self.bot_token = bot_token
        self.default_channel = default_channel
        self.timeout = timeout
    
    def send_message(
        self,
        text: str,
        channel: Optional[str] = None,
        username: str = "AI Agent",
        icon_emoji: str = ":robot_face:",
    ) -> ToolResult:
        """
        Send a simple text message to Slack.
        
        Args:
            text: Message text (supports Slack markdown).
            channel: Target channel (uses default if not specified).
            username: Display name for the message.
            icon_emoji: Emoji icon for the message.
            
        Returns:
            ToolResult indicating success.
            
        Example:
            >>> tool.send_message("Pipeline completed!", channel="#ml-alerts")
        """
        channel = channel or self.default_channel
        
        payload = {
            "text": text,
            "channel": channel,
            "username": username,
            "icon_emoji": icon_emoji,
        }
        
        return self._send(payload)
    
    def post_report(
        self,
        title: str,
        sections: List[Dict[str, str]],
        channel: Optional[str] = None,
        color: str = "#36a64f",
        footer: Optional[str] = None,
    ) -> ToolResult:
        """
        Post a formatted report to Slack.
        
        Args:
            title: Report title.
            sections: List of {"title": str, "value": str} dicts.
            channel: Target channel.
            color: Attachment color (hex).
            footer: Optional footer text.
            
        Returns:
            ToolResult indicating success.
            
        Example:
            >>> tool.post_report(
            ...     title="Model Drift Report",
            ...     sections=[
            ...         {"title": "Drift Score", "value": "0.15"},
            ...         {"title": "Action", "value": "Retraining triggered"},
            ...     ]
            ... )
        """
        channel = channel or self.default_channel
        
        fields = [
            {"title": s["title"], "value": s["value"], "short": True}
            for s in sections
        ]
        
        attachment = {
            "color": color,
            "title": title,
            "fields": fields,
            "footer": footer or f"AI Agent • {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "ts": datetime.now().timestamp(),
        }
        
        payload = {
            "channel": channel,
            "attachments": [attachment],
        }
        
        return self._send(payload)
    
    def post_alert(
        self,
        title: str,
        message: str,
        severity: str = "warning",
        channel: Optional[str] = None,
    ) -> ToolResult:
        """
        Post an alert message with severity indicator.
        
        Args:
            title: Alert title.
            message: Alert details.
            severity: One of "info", "warning", "error", "critical".
            channel: Target channel.
            
        Returns:
            ToolResult indicating success.
        """
        colors = {
            "info": "#36a64f",
            "warning": "#f2c744",
            "error": "#e01e5a",
            "critical": "#ff0000",
        }
        
        emojis = {
            "info": ":information_source:",
            "warning": ":warning:",
            "error": ":x:",
            "critical": ":rotating_light:",
        }
        
        color = colors.get(severity, colors["warning"])
        emoji = emojis.get(severity, emojis["warning"])
        
        return self.post_report(
            title=f"{emoji} {title}",
            sections=[{"title": "Details", "value": message}],
            channel=channel,
            color=color,
        )
    
    def _send(self, payload: Dict) -> ToolResult:
        """Send payload to Slack."""
        try:
            if self.webhook_url:
                response = requests.post(
                    self.webhook_url,
                    json=payload,
                    timeout=self.timeout,
                )
                
                if response.status_code == 200:
                    return ToolResult(
                        success=True,
                        data={"channel": payload.get("channel"), "sent": True},
                    )
                else:
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"Slack error: {response.status_code} - {response.text}",
                    )
            
            elif self.bot_token:
                response = requests.post(
                    "https://slack.com/api/chat.postMessage",
                    headers={"Authorization": f"Bearer {self.bot_token}"},
                    json=payload,
                    timeout=self.timeout,
                )
                
                data = response.json()
                if data.get("ok"):
                    return ToolResult(
                        success=True,
                        data={"channel": data.get("channel"), "ts": data.get("ts")},
                    )
                else:
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"Slack API error: {data.get('error')}",
                    )
            
            else:
                # Mock mode - just log the message
                logger.info(f"[MOCK SLACK] {payload}")
                return ToolResult(
                    success=True,
                    data={"mock": True, "payload": payload},
                )
                
        except requests.RequestException as e:
            logger.error(f"Slack error: {e}")
            return ToolResult(success=False, data=None, error=str(e))


# Convenience functions
_default_tool: Optional[SlackTool] = None


def get_tool(
    webhook_url: Optional[str] = None,
    bot_token: Optional[str] = None,
) -> SlackTool:
    """Get or create default Slack tool instance."""
    global _default_tool
    if _default_tool is None:
        _default_tool = SlackTool(webhook_url=webhook_url, bot_token=bot_token)
    return _default_tool


def send_message(text: str, channel: Optional[str] = None) -> ToolResult:
    """Send a message. See SlackTool.send_message for details."""
    return get_tool().send_message(text, channel)


def post_report(
    title: str,
    sections: List[Dict[str, str]],
    channel: Optional[str] = None,
) -> ToolResult:
    """Post a report. See SlackTool.post_report for details."""
    return get_tool().post_report(title, sections, channel)
