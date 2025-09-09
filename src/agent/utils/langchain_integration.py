# enhanced_sap_chatbot.py
from django.http import JsonResponse
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
# from langchain.llms import AzureOpenAI
from langchain_community.llms import AzureOpenAI

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.chat_history import BaseChatMessageHistory
from typing import List, Dict, Optional, Any
import json
import re
from datetime import datetime

from conversation.models.conversation import Conversation
from conversation.models.message import Message
from conversation.models.message_meta import MessageMeta


class DjangoChatMessageHistory(BaseChatMessageHistory):
    """Custom LangChain message history backed by Django models"""
    
    def __init__(self, conversation_uuid: str):
        self.conversation_uuid = conversation_uuid
        self.conversation = None
        try:
            self.conversation = Conversation.active.get(uuid=conversation_uuid)
        except Conversation.DoesNotExist:
            pass

    @property
    def messages(self) -> List[BaseMessage]:
        """Load messages from Django models"""
        if not self.conversation:
            return []
        
        messages = []
        db_messages = (
            Message.active
            .filter(conversation=self.conversation, is_deleted=False)
            .order_by("created_at")
            .values("sender", "text", "ai_model_response")
        )
        
        for msg in db_messages:
            sender = (msg["sender"] or "").lower()
            if sender == "user":
                content = (msg["text"] or "").strip()
                # Trim after business insights as you do in your code
                content = self._trim_after_business_insight(content)
                if content:
                    messages.append(HumanMessage(content=content))
            else:
                content = (msg["ai_model_response"] or msg["text"] or "").strip()
                content = self._trim_after_business_insight(content)
                if content:
                    messages.append(AIMessage(content=content))
        
        return messages

    # def add_user_message(self, message: str) -> None:
    #     """Add user message to Django models"""
    #     if self.conversation:
    #         Message.objects.create(
    #             conversation=self.conversation,
    #             sender="user",
    #             text=message
    #         )

    # def add_ai_message(self, message: str) -> None:
    #     """Add AI message to Django models"""
    #     if self.conversation:
    #         Message.objects.create(
    #             conversation=self.conversation,
    #             sender="bot",
    #             ai_model_response=message
    #         )
    def add_user_message(self, message: str) -> None:
        pass  # Handled by your existing Django logic

    def add_ai_message(self, message: str) -> None:
        pass  # Handled by your existing Django logic

    def clear(self) -> None:
        """Clear conversation history"""
        if self.conversation:
            Message.active.filter(conversation=self.conversation).update(is_deleted=True)

    def _trim_after_business_insight(self, text: str) -> str:
        """Same logic as your existing code"""
        if not text:
            return ""
        pattern = re.compile(
            r'(?im)^[ \t]{0,3}(?:#{1,6}[ \t]*)?(?:\*\*|__)?[ \t]*Business[ \t]+Insights?(?:\*\*|__)?[ \t]*:?[ \t]*$'
        )
        match = pattern.search(text)
        return text[:match.start()].rstrip() if match else text

_conversation_memories: Dict[str, ConversationSummaryBufferMemory] = {}

def get_or_create_langchain_memory(conversation_uuid: str, llm) -> ConversationSummaryBufferMemory:
    """Get or create LangChain memory for a conversation"""
    if conversation_uuid not in _conversation_memories:
        message_history = DjangoChatMessageHistory(conversation_uuid)
        
        memory = ConversationSummaryBufferMemory(
            llm=llm,
            chat_memory=message_history,
            max_token_limit=4000,
            return_messages=True,
            memory_key="chat_history"
        )
        
        _conversation_memories[conversation_uuid] = memory
    
    return _conversation_memories[conversation_uuid]

class EnhancedSAPChatbot:
    """Enhanced SAP Sales Chatbot with LangChain memory integration"""
    
    def __init__(self, azure_openai_config: Dict[str, str], adx_config: Dict[str, str]):
        # Initialize Azure OpenAI
        self.llm = AzureOpenAI(
            deployment_name=azure_openai_config['deployment_name'],
            openai_api_base=azure_openai_config['api_base'],
            openai_api_key=azure_openai_config['api_key'],
            openai_api_version=azure_openai_config['api_version']
        )
        
        self.adx_config = adx_config
        self._conversation_memories: Dict[str, ConversationSummaryBufferMemory] = {}

    def get_or_create_memory(self, conversation_uuid: str) -> ConversationSummaryBufferMemory:
        """Get or create LangChain memory for a conversation"""
        if conversation_uuid not in self._conversation_memories:
            # Create custom message history backed by Django
            message_history = DjangoChatMessageHistory(conversation_uuid)
            
            # Create LangChain memory with Django backend
            memory = ConversationSummaryBufferMemory(
                llm=self.llm,
                chat_memory=message_history,
                max_token_limit=4000,  # Adjust based on your model's context
                return_messages=True,
                memory_key="chat_history"
            )
            
            self._conversation_memories[conversation_uuid] = memory
        
        return self._conversation_memories[conversation_uuid]

    def generate_kql_with_langchain(self, user_req: str, conversation_uuid: Optional[str] = None, strict=False) -> str:
        """Enhanced KQL generation using LangChain memory"""
        
        # Get your existing prompt base
        prompt = self._build_base_kql_prompt()
        prompt += self._build_schema_prompt_block()
        prompt += self._build_output_format_block()
        
        # Get conversation memory
        memory = None
        conversation_context = ""
        if conversation_uuid:
            memory = self.get_or_create_memory(conversation_uuid)
            
            # Get formatted conversation history from LangChain
            messages = memory.chat_memory.messages[-40:]  # Last 20 exchanges
            conversation_context = self._format_langchain_messages(messages)
        
        # Add conversation context to prompt
        if conversation_context:
            prompt += f"\n\nCONVERSATION HISTORY:\n{conversation_context}"
            prompt += self._build_context_handling_instructions()
        
        # Get your existing metadata (keep your current logic)
        meta_data = get_latest_meta(conversation_uuid) if conversation_uuid else []
        meta_block = ""
        for meta in meta_data:
            if meta.meta_json:
                meta_block += f"\n\n{json.dumps(meta.meta_json)}"
        
        if meta_block:
            prompt += f"\n\nPrevious Context Metadata: {meta_block}"
        
        # Add your existing user access scope logic
        prompt += self._add_user_scope_logic()
        
        # Add conditional instructions (MTD, YTD, etc.) - keep your existing logic
        prompt += self._add_conditional_instructions(user_req)
        
        prompt += f"\n\nUser request: {user_req}"
        
        # Generate KQL using Azure OpenAI
        response = self.llm.invoke([{"role": "user", "content": prompt}]).content
        
        # Extract and clean KQL (keep your existing logic)
        meta, kql_body = self._extract_meta_line_and_strip(response)
        kql_clean = self._clean_kql(kql_body)
        
        # Save metadata (keep your existing logic)
        if conversation_uuid:
            message_id = get_latest_message_id(conversation_uuid)
            save_meta(conversation_uuid, message_id, meta)
        
        return kql_clean

    def handle_user_query_enhanced(self, user_prompt: str, conversation_id: str = None) -> str:
        """Enhanced version of your handle_user_query with LangChain integration"""
        
        # Keep your existing general query check
        if not self._is_sales_analysis_query(user_prompt):
            return self._handle_general_query(user_prompt, conversation_id)
        
        # Keep your existing date detection logic
        start_date, end_date = self._detect_date_filter_using_llm(user_prompt)
        if start_date and end_date:
            start_date_str = start_date.strftime("%Y-%m-%d")
            end_date_str = end_date.strftime("%Y-%m-%d")
            user_prompt += f" from {start_date_str} to {end_date_str}"
        
        # Generate KQL with LangChain memory
        kql = self.generate_kql_with_langchain(user_prompt, conversation_id)
        
        # Keep your existing KQL formatting and territory mapping
        kql = self._format_dates(kql)
        kql = re.sub(r'ago\(3mo\)', 'ago(90d)', kql, flags=re.I)
        kql = re.sub(r'startofquarter\((.*?)\)', r'startofmonth(\1)', kql, flags=re.I)
        kql = self._apply_territory_mapping(kql, user_prompt)
        
        # Execute with retry (keep your existing logic)
        for attempt in (1, 2):
            try:
                cols, rows = self._execute_adx_query(kql)
                break
            except Exception as e:
                if attempt == 1:
                    kql = self.generate_kql_with_langchain(user_prompt, conversation_id, strict=True)
                    continue
                return "Please refine your query for better results. I'm learning day by day and will help you improve your query."
        
        if not rows:
            return "No data found matching your criteria. Please refine your query for more specific results."
        
        # Format results (keep your existing logic)
        result_data = self._format_query_results(cols, rows)
        result_json = json.dumps(result_data, default=str, indent=2)
        
        # Build response using LangChain memory context
        response = self._build_response_with_memory(
            user_prompt, conversation_id, result_json, kql
        )
        
        # Save conversation turn to LangChain memory
        if conversation_id:
            memory = self.get_or_create_memory(conversation_id)
            memory.save_context(
                {"input": user_prompt},
                {"output": response}
            )
        
        return response

    def _format_langchain_messages(self, messages: List[BaseMessage]) -> str:
        """Format LangChain messages for prompt inclusion"""
        if not messages:
            return "No previous conversation."
        
        formatted = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted.append(f"Assistant: {msg.content}")
        
        return "\n".join(formatted)

    def _build_response_with_memory(self, user_prompt: str, conversation_id: str, 
                                   result_json: str, kql: str) -> str:
        """Build response considering conversation memory"""
        
        memory = self.get_or_create_memory(conversation_id) if conversation_id else None
        conversation_context = ""
        
        if memory:
            messages = memory.chat_memory.messages[-10:]  # Last 5 exchanges for context
            conversation_context = self._format_langchain_messages(messages)
        
        # Enhanced response prompt with conversation context
        response_prompt = f"""
You are a SAP Sales Analysis Assistant with access to conversation history.

CONVERSATION CONTEXT:
{conversation_context}

CURRENT USER QUERY: {user_prompt}

QUERY RESULTS: {result_json}

INSTRUCTIONS:
- Provide a natural, conversational response that builds on previous exchanges
- Reference previous discussions when relevant (e.g., "As we discussed earlier...")
- If user asks follow-up questions, connect to previous context
- Format numerical results clearly with bullet points
- Provide business insights and suggestions
- Use "Depo/Sales Office" for gsber values
- Amounts are in BDT

Response:
"""
        
        response = self.llm.invoke([{"role": "user", "content": response_prompt}]).content
        return response

    def get_conversation_summary(self, conversation_uuid: str) -> str:
        """Get conversation summary from LangChain memory"""
        memory = self.get_or_create_memory(conversation_uuid)
        if hasattr(memory, 'moving_summary_buffer'):
            return memory.moving_summary_buffer
        return "No conversation summary available."

    def clear_conversation_memory(self, conversation_uuid: str) -> None:
        """Clear both LangChain memory and Django records"""
        if conversation_uuid in self._conversation_memories:
            self._conversation_memories[conversation_uuid].clear()
            del self._conversation_memories[conversation_uuid]
        
        # Also clear Django records if needed
        try:
            conversation = Conversation.active.get(uuid=conversation_uuid)
            Message.active.filter(conversation=conversation).update(is_deleted=True)
        except Conversation.DoesNotExist:
            pass

    # Keep all your existing helper methods
    def _build_base_kql_prompt(self) -> str:
        """Your existing SYSTEM_PROMPT_KQL"""
        return """You are an expert KQL generator for SAP sales data..."""
    
    def _build_schema_prompt_block(self) -> str:
        """Your existing build_schema_prompt_block()"""
        # Keep your existing implementation
        pass
    
    def _build_output_format_block(self) -> str:
        """Your existing output format instructions"""
        return """
    OUTPUT FORMAT (must follow exactly):
    - First line MUST be a one-line comment with compact JSON, then raw KQL only:
    // META {"dates":{"start":"YYYY-MM-DD","end":"YYYY-MM-DD"},"filters":{"<column>":["<v1>","<v2>"]}}
    """
    
    def _build_context_handling_instructions(self) -> str:
        """Enhanced context handling with LangChain awareness"""
        return """
    CONVERSATION CONTEXT HANDLING:
    - Use conversation history to understand references like "that region", "same period", "those customers"
    - If user asks follow-up questions, maintain context from previous exchanges
    - When filters are not specified in current request, intelligently inherit from conversation context
    - If user says "compare with last month" or similar, use conversation history to understand the baseline
    - Override previous context only when explicitly requested or new values provided
    - Add applied context as comments in generated KQL
    """

    # Keep all your existing methods with minimal changes
    def _is_sales_analysis_query(self, prompt: str) -> bool:
        """Your existing is_sales_analysis_query logic"""
        pass
    
    def _detect_date_filter_using_llm(self, prompt: str):
        """Your existing detect_date_filter_using_llm logic"""
        pass
    
    def _execute_adx_query(self, kql: str):
        """Your existing ADX execution logic"""
        pass
    
    def _format_query_results(self, cols, rows) -> List[Dict]:
        """Your existing result formatting logic"""
        pass


# Integration with your existing Django views
class SAPChatbotService:
    """Service layer to integrate with your existing Django architecture"""
    
    def __init__(self):
        self.chatbot = None
        self._initialize_chatbot()
    
    def _initialize_chatbot(self):
        """Initialize chatbot with your existing config"""
        azure_config = {
            'deployment_name': 'your-deployment',
            'api_base': 'your-api-base',
            'api_key': 'your-api-key',
            'api_version': '2023-05-15'
        }
        
        adx_config = {
            # Your existing ADX config
        }
        
        self.chatbot = EnhancedSAPChatbot(azure_config, adx_config)

    def process_user_message(self, user_prompt: str, conversation_uuid: str = None) -> str:
        """Main entry point - replaces your handle_user_query"""
        return self.chatbot.handle_user_query_enhanced(user_prompt, conversation_uuid)

    def get_conversation_insights(self, conversation_uuid: str) -> Dict[str, Any]:
        """Get conversation insights and patterns"""
        memory = self.chatbot.get_or_create_memory(conversation_uuid)
        
        # Analyze conversation patterns
        messages = memory.chat_memory.messages
        user_messages = [m.content for m in messages if isinstance(m, HumanMessage)]
        
        return {
            "total_exchanges": len(messages) // 2,
            "conversation_summary": self.chatbot.get_conversation_summary(conversation_uuid),
            "recent_topics": self._extract_recent_topics(user_messages[-5:]),
            "common_filters": self._extract_common_filters(conversation_uuid)
        }

    def _extract_recent_topics(self, recent_messages: List[str]) -> List[str]:
        """Extract topics from recent messages"""
        topics = []
        keywords = ["revenue", "sales", "growth", "region", "dealer", "product", "trend"]
        
        for msg in recent_messages:
            msg_lower = msg.lower()
            found_topics = [kw for kw in keywords if kw in msg_lower]
            topics.extend(found_topics)
        
        return list(set(topics))

    def _extract_common_filters(self, conversation_uuid: str) -> Dict[str, Any]:
        """Extract commonly used filters from conversation metadata"""
        meta_data = get_latest_meta(conversation_uuid)
        common_filters = {}
        
        for meta in meta_data[-10:]:  # Last 10 metadata entries
            if meta.meta_json and "filters" in meta.meta_json:
                for col, values in meta.meta_json["filters"].items():
                    if col not in common_filters:
                        common_filters[col] = []
                    common_filters[col].extend(values)
        
        # Deduplicate
        for col, values in common_filters.items():
            common_filters[col] = list(set(values))
        
        return common_filters


# Usage in your Django views
def chat_message_view(request):
    """Your existing Django view with LangChain integration"""
    user_prompt = request.POST.get('message', '')
    conversation_uuid = request.POST.get('conversation_id')
    
    # Initialize service
    chatbot_service = SAPChatbotService()
    
    # Process message with enhanced memory
    response = chatbot_service.process_user_message(user_prompt, conversation_uuid)
    
    # Get conversation insights if needed
    insights = chatbot_service.get_conversation_insights(conversation_uuid)
    
    return JsonResponse({
        'response': response,
        'conversation_insights': insights
    })


# Migration strategy for existing conversations
def migrate_existing_conversations_to_langchain():
    """One-time migration to populate LangChain memory with existing conversations"""
    
    chatbot_service = SAPChatbotService()
    
    for conversation in Conversation.active.all():
        try:
            # Get existing messages
            messages = Message.active.filter(
                conversation=conversation, 
                is_deleted=False
            ).order_by("created_at")
            
            # Create LangChain memory
            memory = chatbot_service.chatbot.get_or_create_memory(str(conversation.uuid))
            
            # Populate memory with existing messages
            for i in range(0, len(messages), 2):
                user_msg = messages[i] if i < len(messages) else None
                bot_msg = messages[i+1] if i+1 < len(messages) else None
                
                if user_msg and user_msg.sender == "user":
                    user_text = user_msg.text or ""
                    bot_text = ""
                    
                    if bot_msg and bot_msg.sender == "bot":
                        bot_text = bot_msg.ai_model_response or bot_msg.text or ""
                    
                    # Save to LangChain memory without duplicating in DB
                    memory.save_context(
                        {"input": user_text},
                        {"output": bot_text}
                    )
            
            print(f"Migrated conversation {conversation.uuid}")
            
        except Exception as e:
            print(f"Error migrating conversation {conversation.uuid}: {e}")