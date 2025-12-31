"""
Telegram RAG Bot - Main Application

A Telegram bot that answers questions using RAG (Retrieval Augmented Generation)
and can describe images using vision models.
"""
import asyncio
import logging
from typing import Dict, List
from collections import defaultdict

from telegram import Update, BotCommand
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from config import config
from rag import Embedder, Retriever, LLMClient

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Global instances
retriever: Retriever = None
llm_client: LLMClient = None

# Message history per user (user_id -> list of messages)
message_history: Dict[int, List[Dict]] = defaultdict(list)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /start command."""
    welcome_message = """🤖 **Welcome to the RAG Bot!**

I'm an AI assistant that can:
• 📚 Answer questions from my knowledge base using `/ask`
• 🖼️ Describe images you send me using `/image`

**Commands:**
• `/ask <your question>` - Ask me anything from the knowledge base
• `/image` - Send an image for description (reply to an image or send after this command)
• `/sources` - Show source documents used in last answer
• `/summarize` - Summarize our recent conversation
• `/help` - Show this help message

Try asking: `/ask What is the remote work policy?`"""
    
    await update.message.reply_text(welcome_message, parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /help command."""
    help_text = """📖 **Bot Commands**

**RAG Queries:**
`/ask <question>` - Ask a question and get an answer from the knowledge base

**Image Description:**
`/image` - Describe an uploaded image
• Reply to an image with `/image`
• Or send `/image` then upload an image

**Utilities:**
`/sources` - Show which documents were used in the last answer
`/summarize` - Get a summary of our recent conversation
`/help` - Show this help message

**Examples:**
• `/ask How many days of annual leave do I get?`
• `/ask What are the password requirements?`
• `/ask Tell me about the product pricing plans`

**Tips:**
• Be specific in your questions for better answers
• The bot remembers your last 3 interactions for context"""
    
    await update.message.reply_text(help_text, parse_mode="Markdown")


async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /ask command for RAG queries."""
    user_id = update.effective_user.id
    
    # Get the query from the command
    if not context.args:
        await update.message.reply_text(
            "❓ Please provide a question after /ask\n"
            "Example: `/ask What is the leave policy?`",
            parse_mode="Markdown"
        )
        return
    
    query = " ".join(context.args)
    
    # Show typing indicator
    await update.message.chat.send_action("typing")
    
    try:
        # Retrieve relevant documents
        results = retriever.search(query, top_k=config.top_k_results)
        
        if not results:
            await update.message.reply_text(
                "🔍 I couldn't find any relevant information in my knowledge base. "
                "Try rephrasing your question."
            )
            return
        
        # Build context from retrieved documents
        context_text = "\n\n---\n\n".join([
            f"[Source: {r['source']}]\n{r['content']}" 
            for r in results
        ])
        
        # Store sources for /sources command
        context.user_data["last_sources"] = results
        
        # Include message history for context
        history_context = ""
        if user_id in message_history and message_history[user_id]:
            history_items = message_history[user_id][-config.max_history_per_user:]
            history_context = "\n\nRecent conversation:\n" + "\n".join([
                f"Q: {h['query']}\nA: {h['answer'][:200]}..." 
                for h in history_items
            ])
        
        full_context = context_text + history_context
        
        # Generate answer using LLM
        answer = await llm_client.generate(query, full_context)
        
        # Store in message history
        message_history[user_id].append({
            "query": query,
            "answer": answer,
            "sources": [r["source"] for r in results]
        })
        
        # Keep only last N messages
        if len(message_history[user_id]) > config.max_history_per_user:
            message_history[user_id] = message_history[user_id][-config.max_history_per_user:]
        
        # Format response with source indicators
        sources_list = list(set(r["source"] for r in results))
        sources_text = ", ".join(f"`{s}`" for s in sources_list)
        
        response = f"📝 **Answer:**\n\n{answer}\n\n📚 *Sources: {sources_text}*"
        
        await update.message.reply_text(response, parse_mode="Markdown")
        
    except Exception as e:
        logger.error(f"Error processing ask command: {e}")
        await update.message.reply_text(
            "❌ Sorry, I encountered an error processing your question. Please try again."
        )


async def sources(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /sources command to show last used sources."""
    last_sources = context.user_data.get("last_sources", [])
    
    if not last_sources:
        await update.message.reply_text(
            "📚 No sources available. Ask a question first with `/ask`",
            parse_mode="Markdown"
        )
        return
    
    response = "📚 Sources from last answer:\n\n"
    for i, source in enumerate(last_sources, 1):
        relevance = int(source["relevance_score"] * 100)
        snippet = source["content"][:150].replace("\n", " ") + "..."
        response += f"{i}. {source['source']} (Relevance: {relevance}%)\n"
        response += f"   {snippet}\n\n"
    
    await update.message.reply_text(response)


async def image_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /image command for image description."""
    # Check if replying to a photo
    if update.message.reply_to_message and update.message.reply_to_message.photo:
        await process_image(update, context, update.message.reply_to_message.photo[-1])
    else:
        # Set flag to process next image
        context.user_data["waiting_for_image"] = True
        await update.message.reply_text(
            "🖼️ Please send me an image and I'll describe it for you!"
        )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming photos."""
    if context.user_data.get("waiting_for_image", False):
        context.user_data["waiting_for_image"] = False
        await process_image(update, context, update.message.photo[-1])
    else:
        await update.message.reply_text(
            "🖼️ Got your image! Use `/image` (reply to this image) to get a description.",
            parse_mode="Markdown"
        )


async def process_image(update: Update, context: ContextTypes.DEFAULT_TYPE, photo) -> None:
    """Process and describe an image."""
    await update.message.chat.send_action("typing")
    
    try:
        # Download the image
        file = await context.bot.get_file(photo.file_id)
        image_bytes = await file.download_as_bytearray()
        
        # Get custom prompt if provided
        custom_prompt = ""
        if context.args:
            custom_prompt = " ".join(context.args)
        
        # Generate description
        description = await llm_client.describe_image(bytes(image_bytes), custom_prompt)
        
        # Store in history
        user_id = update.effective_user.id
        message_history[user_id].append({
            "query": "[Image uploaded]",
            "answer": description,
            "sources": ["image_description"]
        })
        
        await update.message.reply_text(f"🖼️ Image Description:\n\n{description}")
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        await update.message.reply_text(
            "❌ Sorry, I couldn't process that image. Please try again."
        )


async def summarize(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /summarize command."""
    user_id = update.effective_user.id
    
    if user_id not in message_history or not message_history[user_id]:
        await update.message.reply_text(
            "📝 No conversation history to summarize. Start by asking questions with `/ask`",
            parse_mode="Markdown"
        )
        return
    
    await update.message.chat.send_action("typing")
    
    try:
        # Build conversation summary
        history = message_history[user_id]
        conversation = "\n\n".join([
            f"Q: {h['query']}\nA: {h['answer'][:300]}..." 
            for h in history
        ])
        
        summary_prompt = f"""Please provide a brief summary of this conversation:

{conversation}

Summarize the main topics discussed and key information exchanged."""
        
        summary = await llm_client.generate(summary_prompt)
        
        await update.message.reply_text(
            f"📋 **Conversation Summary:**\n\n{summary}",
            parse_mode="Markdown"
        )
        
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        await update.message.reply_text(
            "❌ Sorry, I couldn't generate a summary. Please try again."
        )


async def post_init(application: Application) -> None:
    """Initialize bot after startup."""
    # Set bot commands for the menu
    commands = [
        BotCommand("start", "Start the bot"),
        BotCommand("ask", "Ask a question from knowledge base"),
        BotCommand("image", "Describe an uploaded image"),
        BotCommand("sources", "Show sources from last answer"),
        BotCommand("summarize", "Summarize conversation"),
        BotCommand("help", "Show help message"),
    ]
    await application.bot.set_my_commands(commands)
    logger.info("Bot commands set successfully")


def initialize_rag():
    """Initialize RAG components."""
    global retriever, llm_client
    
    logger.info("Initializing RAG components...")
    
    # Initialize embedder and retriever
    embedder = Embedder(model_name=config.embedding_model)
    retriever = Retriever(
        collection_name=config.collection_name,
        persist_directory=config.db_directory,
        embedder=embedder
    )
    
    # Index documents if needed
    if retriever.get_document_count() == 0:
        logger.info("Indexing documents...")
        count = retriever.index_documents(data_dir=config.data_directory)
        logger.info(f"Indexed {count} document chunks")
    else:
        logger.info(f"Using existing index with {retriever.get_document_count()} documents")
    
    # Initialize LLM client
    if config.llm_provider == "openai":
        llm_client = LLMClient(
            provider="openai",
            api_key=config.openai_api_key,
            model=config.openai_model
        )
    else:
        llm_client = LLMClient(
            provider="ollama",
            model=config.ollama_model,
            vision_model=config.ollama_vision_model,
            host=config.ollama_host
        )
    
    logger.info(f"LLM client initialized with provider: {config.llm_provider}")


def main():
    """Main function to run the bot."""
    # Validate configuration
    config.validate()
    
    # Initialize RAG components
    initialize_rag()
    
    # Create the Application
    application = (
        Application.builder()
        .token(config.telegram_token)
        .post_init(post_init)
        .build()
    )
    
    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("ask", ask))
    application.add_handler(CommandHandler("image", image_command))
    application.add_handler(CommandHandler("sources", sources))
    application.add_handler(CommandHandler("summarize", summarize))
    
    # Add photo handler
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    
    # Run the bot
    logger.info("Starting Telegram bot...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()



