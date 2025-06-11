use clap::{Parser, ValueHint};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::{self, Write, BufRead};
use std::path::{Path, PathBuf};
use std::process::Command;
use chrono::prelude::*;
use sys_info::hostname;
use dirs;
use async_openai::{
    types::{ChatCompletionRequestMessage, ChatCompletionTool, ChatCompletionToolType, FunctionObject, ChatCompletionRequestSystemMessage},
    config::{OpenAIConfig},
    Client,
};

const CONVERSATION_FILE: &str = "/tmp/ai_conversation";
const LOG_FILE: &str = "/tmp/ai_log.csv";
const DEFAULT_API_BASE: &str = "http://ai3:8080/v1";
const DEFAULT_API_KEY: &str = "empty";
const SYSTEM_PROMPT: &str = r#"
You are an AI assistant that can run terminal commands.
You will use a provided tool to execute terminal commands when necessary.
You may also simply reply to the user without using a tool call.

Important Guidelines:
1. For file editing: Either use `sed` directly OR use `cat -n` to inspect with line numbers and then use `sed` to modify specific lines.
2. Avoid interactive commands unless necessary - most operations should be scriptable.
3. The terminal tool maintains state (like current directory) between calls during the same conversation.
4. Be cautious with destructive operations and always have rollbacks where possible.

Tool: 'terminal'
Purpose: Execute a single terminal command. Maintains current working directory across calls.
Parameters:
  - command: The command string to execute (required)
Special handling for 'cd': If the command starts with 'cd', it will update the working directory without requiring a new prompt.

Output will be captured and returned as a string.

Do not chain commands with '&&' or ';'. Only one command per tool call.
"#;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct State {
    messages: Vec<ChatCompletionRequestMessage>,
    terminal_state: TerminalState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TerminalState {
    cwd: PathBuf,
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

    /// Enable tool/function calling (may not be supported by all models)
    #[arg(short, long)]
    tools: bool,

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
            messages: vec![ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                content: async_openai::types::ChatCompletionRequestSystemMessageContent::Text(SYSTEM_PROMPT.trim().to_owned()),
                name: None,
            })],
            terminal_state: TerminalState {
                cwd: std::env::current_dir()?,
            },
        }
    };

    state.messages.push(ChatCompletionRequestMessage::User(user_message.clone().into()));

    log_event("user", None, &user_message)?;

    // Create async-openai client with config
    let ai_config = OpenAIConfig::new()
        .with_api_base(&cli_args.api_base)
        .with_api_key(&cli_args.api_key);
    let ai_client = Client::with_config(ai_config);

    // Define the terminal tool for API requests
    let terminal_tool = ChatCompletionTool {
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

    loop {
        // Convert state.messages to owned so we can use in request
        let messages = state.messages.clone();
        let mut request = async_openai::types::CreateChatCompletionRequest {
            model: cli_args.model.clone(),
            messages,
            ..Default::default()
        };

        if cli_args.tools {
            request.tools = Some(vec![terminal_tool.clone()]);
        }

        let request = ai_client
            .chat()
            .create(request)
            .await?;

        let choice = request
            .choices
            .first()
            .ok_or(anyhow::anyhow!("No choices returned"))?;
        let message = choice.message.clone();

        if let Some(tool_calls) = message.tool_calls.clone() {
            log_event("assistant_tool_calls", None, &serde_json::to_string(&tool_calls)?)?;
            for call in tool_calls {
                if call.function.name == "terminal" {
                    let function = call.function.clone();
                    let args: HashMap<String, String> = serde_json::from_str(&function.arguments)?;
                    let cmd = args.get("command").ok_or(anyhow::anyhow!("Missing command"))?;

                    if cli_args.safe {
                        println!("Execute command? [y极/N]\n  {}", cmd);
                        let mut r = String::new();
                        io::stdin().read_line(&mut r)?;
                        if r.trim().to_lowercase() != "y" {
                            state.messages.push(ChatCompletionRequestMessage::Tool(
                                async_openai::types::ChatCompletionRequestToolMessage {
                                    content: async_openai::types::ChatCompletionRequestToolMessageContent::Text(format!("Command execution canceled: {}", cmd)),
                                    tool_call_id: call.id.clone(),
                                }
                            ));
                            log_event("tool_canceled", None, cmd)?;
                            continue;
                        }
                    }

                    let result = run_terminal_command(cmd, &mut state)?;
                    state.messages.push(ChatCompletionRequestMessage::Tool(
                        async_openai::types::ChatCompletionRequestToolMessage {
                            content: async_openai::types::ChatCompletionRequestToolMessageContent::Text(result.clone()),
                            tool_call_id: call.id.clone(),
                        }
                    ));
                    log_event("tool_executed", None, &result)?;
                }
            }
        } else {
            let content: &str = message.content.as_deref().unwrap_or_default();
            println!("assistant: {}", content);
            log_event("assistant", None, content)?;
            // Manually create an assistant message from the response
            state.messages.push(ChatCompletionRequestMessage::Assistant(
                async_openai::types::ChatCompletionRequestAssistantMessage {
                    content: message.content.map(|c| async_openai::types::ChatCompletionRequestAssistantMessageContent::Text(c)),
                    name: None,
                    tool_calls: None,
                    function_call: None,
                    audio: None,
                    refusal: None,
                }
            ));
            break;
        }
    }

    save_state(&state)?;
    Ok(())
}

fn run_terminal_command(cmd: &str, state: &mut State) -> anyhow::Result<String> {
    let mut parts = cmd.split_whitespace();
    let main_cmd = parts.next().unwrap();

    if main_cmd == "cd" {
        let dir = parts.next().unwrap_or("~");
        let path = if dir == "~" {
            dirs::home_dir().ok_or(anyhow::anyhow!("No home dir found"))?
        } else {
            let abs_path = if Path::new(dir).is_absolute() {
                PathBuf::from(dir)
            } else {
                state.terminal_state.cwd.join(dir)
            };
            if !abs_path.exists() {
                return Err(anyhow::anyhow!("Directory not found: {:?}", abs_path));
            }
            abs_path
        };
        state.terminal_state.cwd = path.clone();
        return Ok(format!("Changed directory to: {:?}", path));
    }

    let output = Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .current_dir(&state.terminal_state.cwd)
        .output()?;

    let result = if output.status.success() {
        String::from_utf8(output.stdout)?
    } else {
        String::from_utf8(output.stderr)?
    };

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

    wtr.write_record(&[
        &timestamp,
        event_type,
        &host,
        &id,
        &function,
        details,
    ])?;

    wtr.flush()?;
    Ok(())
}

fn save_state(state: &State) -> anyhow::Result<()> {
    let json = serde_json::to_string(state)?;
    fs::write(CONVERSATION_FILE, json)?;
    Ok(())
}
