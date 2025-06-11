use async_openai::{
    Client,
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage, ChatCompletionTool,
        ChatCompletionToolType, FunctionObject,
    },
};
use chrono::prelude::*;
use clap::{Parser, ValueHint};
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::process::Command;
use sys_info::hostname;

const CONVERSATION_FILE: &str = "/tmp/ai_conversation";
const LOG_FILE: &str = "/tmp/ai_log.csv";
const DEFAULT_API_BASE: &str = "http://ai3:8080/v1";
const DEFAULT_API_KEY: &str = "empty";
const SYSTEM_PROMPT: &str = r#"
You are an AI assistant that can run terminal commands.
To execute commands, format them like this:

terminal_call:
```
your commands here
```

Important Guidelines:
1. Commands will be executed as a POSIX-compliant shell script using /bin/sh
2. Each execution starts fresh in the initial working directory
3. You can use full shell syntax including &&, ||, ;, | etc.
4. For file editing: Either use `sed` directly OR use `cat -n` to inspect with line numbers and then use `sed` to modify specific lines.
5. Be cautious with destructive operations and always have rollbacks where possible.

The script will be executed directly from memory when possible, or saved to /tmp/ai_cli_script.sh if needed.
"#;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct State {
    messages: Vec<ChatCompletionRequestMessage>,
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Continue previous conversation
    #[arg(short, long)]
    continue_conversation: bool,

    /// Confirm before executing commands
    #[arg(short, long)]
    safe: bool,

    /// API base URL
    #[arg(short='b', long, default_value_t = DEFAULT_API_BASE.to_string(), value_hint = ValueHint::Url)]
    api_base: String,

    /// API key
    #[arg(short='k', long, default_value_t = DEFAULT_API_KEY.to_string())]
    api_key: String,

    /// Model to use for completions
    #[arg(short, long, default_value_t = String::from("qwen_coder"))]
    model: String,

    message: Vec<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli_args = Args::parse();
    let user_message = if cli_args.message.is_empty() {
        eprint!("Message: ");
        io::stdout().flush()?;
        let mut buffer = String::new();
        io::stdin().lock().read_line(&mut buffer)?;
        buffer.trim().to_owned()
    } else {
        cli_args.message.join(" ")
    };

    let mut state = if cli_args.continue_conversation && Path::new(CONVERSATION_FILE).exists() {
        let s = fs::read_to_string(CONVERSATION_FILE)?;
        serde_json::from_str(&s)?
    } else {
        State {
            messages: vec![ChatCompletionRequestMessage::System(
                ChatCompletionRequestSystemMessage {
                    content: async_openai::types::ChatCompletionRequestSystemMessageContent::Text(
                        SYSTEM_PROMPT.trim().to_owned(),
                    ),
                    name: None,
                },
            )],
        }
    };

    state.messages.push(ChatCompletionRequestMessage::User(
        user_message.clone().into(),
    ));

    log_event("user", None, &user_message)?;

    // Create async-openai client with config
    let ai_config = OpenAIConfig::new()
        .with_api_base(&cli_args.api_base)
        .with_api_key(&cli_args.api_key);
    let ai_client = Client::with_config(ai_config);

    // Define the terminal tool for API requests
    let _terminal_tool = ChatCompletionTool {
        r#type: ChatCompletionToolType::Function,
        function: FunctionObject {
            name: "terminal".to_string(),
            description: Some("Run a terminal command and get the output. Maintains current working directory across calls.".to_string()),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to execute"
                    }
                },
                "required": ["command"]
            })),
            strict: None,
        },
    };

    let messages = state.messages.clone();
    let request = async_openai::types::CreateChatCompletionRequest {
        model: cli_args.model.clone(),
        messages,
        ..Default::default()
    };

    let response = ai_client.chat().create(request).await?;
    let message = response
        .choices
        .first()
        .ok_or(anyhow::anyhow!("No choices returned"))?
        .message
        .clone();
    let content = message.content.as_deref().unwrap_or_default();

    // Print assistant message
    println!("{}", content);
    log_event("assistant", None, content)?;

    // Check for terminal call
    if let Some(script) = extract_terminal_call(content) {
        // Calculate context length
        let context_len = estimate_context_length(&state.messages);
        println!("Current context length: {} tokens", context_len);

        if cli_args.safe {
            print!("Execute script? [Y/n]: ");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            if input.trim().to_lowercase() == "n" {
                log_event("script_canceled", None, "User canceled")?;
                return Ok(());
            }
        }

        let result = run_script(&script)?;
        println!("{}", result);
        log_event("script_output", None, &result)?;

        // Append output to conversation
        state.messages.push(ChatCompletionRequestMessage::Assistant(
            async_openai::types::ChatCompletionRequestAssistantMessage {
                content: Some(
                    async_openai::types::ChatCompletionRequestAssistantMessageContent::Text(
                        format!(
                            "Script executed:\n```\n{}\n```\nOutput:\n{}",
                            script, result
                        ),
                    ),
                ),
                name: None,
                tool_calls: None,
                #[allow(deprecated)]
                function_call: None,
                audio: None,
                refusal: None,
            },
        ));
    }

    save_state(&state)?;
    Ok(())
}

fn extract_terminal_call(content: &str) -> Option<String> {
    let pattern = "terminal_call:\n```\n";
    if let Some(start) = content.find(pattern) {
        let remaining = &content[start + pattern.len()..];
        if let Some(end) = remaining.find("\n```") {
            return Some(remaining[..end].to_string());
        }
    }
    None
}

fn estimate_context_length(messages: &[ChatCompletionRequestMessage]) -> usize {
    let total_words: usize = messages
        .iter()
        .map(|msg| match msg {
            ChatCompletionRequestMessage::System(s) => match &s.content {
                async_openai::types::ChatCompletionRequestSystemMessageContent::Text(t) => {
                    t.split_whitespace().count()
                }
                _ => 0,
            },
            ChatCompletionRequestMessage::User(u) => match &u.content {
                async_openai::types::ChatCompletionRequestUserMessageContent::Text(t) => {
                    t.split_whitespace().count()
                }
                _ => 0,
            },
            ChatCompletionRequestMessage::Assistant(a) => {
                a.content.as_ref().map_or(0, |c| match c {
                    async_openai::types::ChatCompletionRequestAssistantMessageContent::Text(t) => {
                        t.split_whitespace().count()
                    }
                    _ => 0,
                })
            }
            _ => 0,
        })
        .sum();
    (total_words as f64 * 1.2).round() as usize
}

fn run_script(script: &str) -> anyhow::Result<String> {
    let output = Command::new("sh")
        .arg("-c")
        .arg(script)
        .current_dir(std::env::current_dir()?)
        .output()?;

    let result = String::from_utf8(output.stdout)?;
    if !output.status.success() {
        let stderr = String::from_utf8(output.stderr)?;
        return Ok(format!("Error:\n{}\nOutput:\n{}", stderr, result));
    }
    Ok(result)
}

#[allow(unused_variables)]
fn log_event(
    event_type: &str,
    tool_call: Option<&async_openai::types::ChatCompletionMessageToolCall>,
    details: &str,
) -> anyhow::Result<()> {
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(LOG_FILE)?;

    let host = hostname().unwrap_or_else(|_| "unknown".to_string());
    let timestamp = Utc::now().to_rfc3339();

    let (function, id) = match tool_call {
        Some(call) => (call.function.name.clone(), call.id.clone()),
        None => ("".to_string(), "".to_string()),
    };

    let mut wtr = csv::WriterBuilder::new()
        .has_headers(!Path::new(LOG_FILE).exists())
        .from_writer(file);

    wtr.write_record(&[&timestamp, event_type, &host, &id, &function, details])?;

    wtr.flush()?;
    Ok(())
}

fn save_state(state: &State) -> anyhow::Result<()> {
    let json = serde_json::to_string(state)?;
    fs::write(CONVERSATION_FILE, json)?;
    Ok(())
}
