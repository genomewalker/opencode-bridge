//! fp — token-efficient file patcher for Claude Code
//!
//! Modes:
//!   fp <file> <old_str> <new_str>        direct search-replace patch
//!   fp --symbol <file> <symbol> <body>   replace symbol (fn/def/class) by name
//!   fp --hook                            Edit tool hook: stdin JSON → patch + confirm
//!   fp gain                              show token savings stats
//!
//! Build:
//!   rustc fp.rs -o ~/.claude/bin/fp

use std::env;
use std::fs;
use std::io::{self, Read, Write as IoWrite};
use std::path::Path;
use std::process;

// ── Stats file ────────────────────────────────────────────────────────────────

fn stats_path() -> std::path::PathBuf {
    let home = env::var("HOME").unwrap_or_else(|_| ".".into());
    Path::new(&home).join(".claude/mind/fp_stats.json")
}

#[derive(Default)]
struct Stats {
    total_patches: u64,
    tokens_saved: u64,   // estimated: chars_replaced / 4
    tokens_spent: u64,   // actual: confirmation chars / 4
}

impl Stats {
    fn load() -> Self {
        let p = stats_path();
        if !p.exists() { return Self::default(); }
        let content = fs::read_to_string(&p).unwrap_or_default();
        let mut s = Self::default();
        s.total_patches = json_extract_u64(&content, "total_patches");
        s.tokens_saved = json_extract_u64(&content, "tokens_saved");
        s.tokens_spent = json_extract_u64(&content, "tokens_spent");
        s
    }

    fn save(&self) {
        let p = stats_path();
        if let Some(parent) = p.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let json = format!(
            "{{\"total_patches\":{},\"tokens_saved\":{},\"tokens_spent\":{}}}",
            self.total_patches, self.tokens_saved, self.tokens_spent
        );
        let _ = fs::write(&p, json);
    }

    fn record(&mut self, old_chars: usize, new_chars: usize, confirm_len: usize) {
        let replaced = (old_chars + new_chars) as u64;
        self.tokens_saved += replaced / 4;
        self.tokens_spent += confirm_len as u64 / 4;
        self.total_patches += 1;
    }
}

fn json_extract_u64(json: &str, key: &str) -> u64 {
    let needle = format!("\"{}\":", key);
    json.find(&needle)
        .and_then(|i| {
            let rest = json[i + needle.len()..].trim_start();
            rest.split(|c: char| !c.is_ascii_digit()).next()
        })
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

// ── Manifest ──────────────────────────────────────────────────────────────────

fn manifest_path() -> std::path::PathBuf {
    let home = env::var("HOME").unwrap_or_else(|_| ".".into());
    Path::new(&home).join(".claude/mind/fp_manifest.jsonl")
}

fn session_id() -> String {
    env::var("CLAUDE_SESSION_ID").unwrap_or_else(|_| "default".into())
}

fn manifest_record(filepath: &str, kind: &str, line_num: usize) {
    let abs_path = Path::new(filepath)
        .canonicalize()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| filepath.to_string());
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let sid = session_id();
    let entry = format!(
        "{{\"sid\":\"{}\",\"file\":\"{}\",\"type\":\"{}\",\"line\":{},\"ts\":{}}}\n",
        json_escape(&sid), json_escape(&abs_path), kind, line_num, ts
    );
    let p = manifest_path();
    if let Some(parent) = p.parent() { let _ = fs::create_dir_all(parent); }
    if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(&p) {
        let _ = IoWrite::write_all(&mut f, entry.as_bytes());
    }
}

fn manifest_check(filepath: &str) -> Vec<String> {
    let abs_path = Path::new(filepath)
        .canonicalize()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| filepath.to_string());
    let sid = session_id();
    let p = manifest_path();
    if !p.exists() { return vec![]; }
    let content = fs::read_to_string(&p).unwrap_or_default();
    let mut results = Vec::new();
    for line in content.lines() {
        if json_extract_str(line, "sid").as_deref() == Some(&sid)
            && json_extract_str(line, "file").as_deref() == Some(&abs_path)
        {
            let kind = json_extract_str(line, "type").unwrap_or_default();
            let ln = json_extract_u64(line, "line");
            results.push(format!("{} @ L{}", kind, ln));
        }
    }
    results
}

// ── Core patch ────────────────────────────────────────────────────────────────

fn apply_patch(file: &str, old_str: &str, new_str: &str) -> Result<String, String> {
    let path = Path::new(file);
    let fname = path.file_name().and_then(|n| n.to_str()).unwrap_or(file);

    if !path.is_file() {
        return Err(format!("Error: file not found: {}", file));
    }
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Error reading {}: {}", fname, e))?;

    let count = content.matches(old_str).count();
    match count {
        0 => {
            let preview: String = old_str.chars().take(80)
                .map(|c| if c == '\n' { '↵' } else { c }).collect();
            return Err(format!("Error: old_str not found in {}\nSearched for: {:?}", fname, preview));
        }
        n if n > 1 => return Err(format!(
            "Error: old_str matches {} locations in {} — make it more specific", n, fname
        )),
        _ => {}
    }

    let match_start = content.find(old_str).unwrap();
    let line_num = content[..match_start].matches('\n').count() + 1;
    let old_lines = old_str.matches('\n').count() + 1;
    let new_lines = if new_str.is_empty() { 0 } else { new_str.matches('\n').count() + 1 };
    let delta: i64 = new_lines as i64 - old_lines as i64;

    let new_content = content.replacen(old_str, new_str, 1);
    fs::write(path, &new_content)
        .map_err(|e| format!("Error writing {}: {}", fname, e))?;

    let sign = if delta >= 0 { "+" } else { "" };
    let msg = format!("✓ {} patched @ L{} (+{}/-{} lines, net {}{})",
        fname, line_num, new_lines, old_lines, sign, delta);

    let mut stats = Stats::load();
    stats.record(old_str.len(), new_str.len(), msg.len());
    stats.save();
    manifest_record(file, "patch", line_num);

    Ok(msg)
}

// ── Symbol-aware patch ────────────────────────────────────────────────────────

/// Find the byte range [start, end) of a named symbol in source.
/// Supports Python (indent-based) and brace-based languages (Rust, C, JS, Go, etc.)
fn find_symbol_range(content: &str, symbol: &str, ext: &str) -> Option<(usize, usize)> {
    match ext {
        "py" | "pyx" => find_symbol_python(content, symbol),
        _ => find_symbol_brace(content, symbol),
    }
}

fn find_symbol_python(content: &str, symbol: &str) -> Option<(usize, usize)> {
    // Match: def symbol( / def symbol : / class Symbol( / class Symbol:
    let patterns = [
        format!("def {}(", symbol),
        format!("def {}  (", symbol),
        format!("async def {}(", symbol),
        format!("class {}(", symbol),
        format!("class {}:", symbol),
    ];

    let mut def_line_start = None;
    let mut def_indent = 0usize;

    for (byte_pos, line) in line_offsets(content) {
        let trimmed = line.trim_start();
        if patterns.iter().any(|p| trimmed.starts_with(p.as_str())) {
            def_line_start = Some(byte_pos);
            def_indent = line.len() - trimmed.len();
            break;
        }
    }

    let start = def_line_start?;

    // End = next non-blank line at same or lower indentation after first body line
    let mut seen_body = false;
    let mut end = content.len();
    for (byte_pos, line) in line_offsets(content) {
        if byte_pos <= start { continue; }
        if line.trim().is_empty() { continue; }
        let indent = line.len() - line.trim_start().len();
        if seen_body && indent <= def_indent {
            end = byte_pos;
            break;
        }
        seen_body = true;
    }

    Some((start, end))
}

fn find_symbol_brace(content: &str, symbol: &str) -> Option<(usize, usize)> {
    // Match: fn symbol, function symbol, def symbol, class Symbol, impl Symbol, etc.
    let kws = ["fn ", "function ", "async function ", "def ", "class ", "impl ", "struct ", "enum "];

    let mut def_start = None;
    for kw in &kws {
        let needle = format!("{}{}", kw, symbol);
        // Find needle followed by any of: ( < { space \n
        let mut search = content;
        let mut offset = 0;
        while let Some(i) = search.find(&needle) {
            let abs = offset + i;
            let after = &content[abs + needle.len()..];
            let next_char = after.chars().next().unwrap_or(' ');
            if matches!(next_char, '(' | '<' | '{' | ' ' | '\n' | ':') {
                // Make sure it's at a word boundary (not inside a longer name)
                let before_char = if abs > 0 {
                    content[..abs].chars().last().unwrap_or(' ')
                } else { ' ' };
                if !before_char.is_alphanumeric() && before_char != '_' {
                    def_start = Some(abs);
                    break;
                }
            }
            offset += i + 1;
            search = &content[offset..];
        }
        if def_start.is_some() { break; }
    }

    let start = def_start?;

    // Find opening brace, then count to matching close
    let from_start = &content[start..];
    let brace_offset = from_start.find('{')?;
    let abs_brace = start + brace_offset;

    let mut depth = 0i32;
    let mut end = content.len();
    for (i, c) in content[abs_brace..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = abs_brace + i + 1;
                    // Include trailing newline
                    if content[end..].starts_with('\n') { end += 1; }
                    break;
                }
            }
            _ => {}
        }
    }

    Some((start, end))
}

fn line_offsets(content: &str) -> impl Iterator<Item = (usize, &str)> {
    let mut pos = 0;
    content.split('\n').map(move |line| {
        let start = pos;
        pos += line.len() + 1; // +1 for the \n
        (start, line)
    })
}

fn apply_symbol_patch(file: &str, symbol: &str, new_body: &str) -> Result<String, String> {
    let path = Path::new(file);
    let fname = path.file_name().and_then(|n| n.to_str()).unwrap_or(file);
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    if !path.is_file() {
        return Err(format!("Error: file not found: {}", file));
    }
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Error reading {}: {}", fname, e))?;

    let (start, end) = find_symbol_range(&content, symbol, ext)
        .ok_or_else(|| format!("Error: symbol '{}' not found in {}", symbol, fname))?;

    let line_num = content[..start].matches('\n').count() + 1;
    let old_lines = content[start..end].matches('\n').count() + 1;
    let new_body_trimmed = if new_body.ends_with('\n') {
        new_body.to_string()
    } else {
        format!("{}\n", new_body)
    };
    let new_lines = new_body_trimmed.matches('\n').count();
    let delta: i64 = new_lines as i64 - old_lines as i64;

    let new_content = format!("{}{}{}", &content[..start], new_body_trimmed, &content[end..]);
    fs::write(path, &new_content)
        .map_err(|e| format!("Error writing {}: {}", fname, e))?;

    let sign = if delta >= 0 { "+" } else { "" };
    let msg = format!("✓ {}::{} patched @ L{} (+{}/-{} lines, net {}{})",
        fname, symbol, line_num, new_lines, old_lines, sign, delta);

    let mut stats = Stats::load();
    stats.record(end - start, new_body_trimmed.len(), msg.len());
    stats.save();
    manifest_record(file, "symbol", line_num);

    Ok(msg)
}

// ── JSON helpers ──────────────────────────────────────────────────────────────

fn json_extract_str(json: &str, key: &str) -> Option<String> {
    let needle = format!("\"{}\"", key);
    let start = json.find(&needle)?;
    let after_key = &json[start + needle.len()..];
    let colon = after_key.find(':')? + 1;
    let after_colon = after_key[colon..].trim_start();
    if !after_colon.starts_with('"') { return None; }
    let inner = &after_colon[1..];
    let mut result = String::new();
    let mut chars = inner.chars().peekable();
    loop {
        match chars.next()? {
            '"' => break,
            '\\' => match chars.next()? {
                '"' => result.push('"'),
                '\\' => result.push('\\'),
                '/' => result.push('/'),
                'n' => result.push('\n'),
                'r' => result.push('\r'),
                't' => result.push('\t'),
                'u' => {
                    let hex: String = (0..4).filter_map(|_| chars.next()).collect();
                    if let Ok(n) = u32::from_str_radix(&hex, 16) {
                        if let Some(c) = char::from_u32(n) { result.push(c); }
                    }
                }
                c => result.push(c),
            },
            c => result.push(c),
        }
    }
    Some(result)
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

// ── Hook mode ─────────────────────────────────────────────────────────────────

fn hook_mode() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap_or(0);

    let file = json_extract_str(&input, "file_path").unwrap_or_default();
    let old_str = json_extract_str(&input, "old_string").unwrap_or_default();
    let new_str = json_extract_str(&input, "new_string").unwrap_or_default();

    if file.is_empty() || old_str.is_empty() {
        process::exit(0); // not an Edit call we handle — pass through
    }

    let msg = match apply_patch(&file, &old_str, &new_str) {
        Ok(s) => s,
        Err(e) => e,
    };

    // Native tools (Edit) require permissionDecision:block + exit 0, not exit 2.
    // exit 2 is only for Bash tool interception.
    println!(
        "{{\"hookSpecificOutput\":{{\"hookEventName\":\"PreToolUse\",\"permissionDecision\":\"block\",\"additionalContext\":\"{}\"}}}}",
        json_escape(&msg)
    );
    process::exit(0);
}

// ── Gain mode ─────────────────────────────────────────────────────────────────

fn gain_mode() {
    let stats = Stats::load();
    if stats.total_patches == 0 {
        println!("fp: no patches recorded yet.");
        return;
    }
    let saved = stats.tokens_saved;
    let spent = stats.tokens_spent;
    let total = saved + spent;
    let pct = if total > 0 { saved * 100 / total } else { 0 };
    let bar_filled = (pct as usize * 25 / 100).min(25);
    let bar: String = "█".repeat(bar_filled) + &"░".repeat(25 - bar_filled);

    println!("fp Token Savings");
    println!("{}", "═".repeat(44));
    println!("Total patches:    {}", stats.total_patches);
    println!("Tokens saved:     {:>10} ({:.1}%)", saved, pct);
    println!("Tokens spent:     {:>10}", spent);
    println!("Efficiency:       {} {}%", bar, pct);
}

// ── Write hook mode ───────────────────────────────────────────────────────────
// Intercepts Write tool for EXISTING files. New files pass through.

fn write_hook_mode() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap_or(0);

    let file = json_extract_str(&input, "file_path").unwrap_or_default();
    if file.is_empty() { process::exit(0); }

    let path = Path::new(&file);
    if !path.is_file() {
        process::exit(0); // new file — let Write proceed
    }

    let line_count = fs::read_to_string(path)
        .map(|c| c.matches('\n').count())
        .unwrap_or(0);

    if line_count < 50 {
        process::exit(0); // small file — not worth blocking
    }

    // Block the Write and suggest file_patch
    let fname = path.file_name().and_then(|n| n.to_str()).unwrap_or(&file);
    let msg = format!(
        "⚠ Write blocked on existing {}-line file '{}'. Use file_patch or fp for token-efficient edits.",
        line_count, fname
    );
    println!(
        "{{\"hookSpecificOutput\":{{\"hookEventName\":\"PreToolUse\",\"permissionDecision\":\"block\",\"additionalContext\":\"{}\"}}}}",
        json_escape(&msg)
    );
    process::exit(0);
}

// ── Read hook mode ────────────────────────────────────────────────────────────
// Advisory when file was already patched this session — skips re-reading.

fn read_hook_mode() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap_or(0);

    let file = json_extract_str(&input, "file_path").unwrap_or_default();
    if file.is_empty() { process::exit(0); }

    let patches = manifest_check(&file);
    if patches.is_empty() { process::exit(0); }

    let fname = Path::new(&file).file_name()
        .and_then(|n| n.to_str()).unwrap_or(&file);
    let summary = patches.join(", ");
    let msg = format!(
        "[fp] {} was already patched this session: {}. Use file_patch/symbol_patch for further edits, or Read with offset if inspection needed.",
        fname, summary
    );
    println!(
        "{{\"hookSpecificOutput\":{{\"hookEventName\":\"PreToolUse\",\"additionalContext\":\"{}\"}}}}",
        json_escape(&msg)
    );
    process::exit(0);
}

// ── Manifest mode ─────────────────────────────────────────────────────────────

fn manifest_mode() {
    let p = manifest_path();
    if !p.exists() {
        println!("fp manifest: no patches recorded yet.");
        return;
    }
    let sid = session_id();
    let content = fs::read_to_string(&p).unwrap_or_default();
    let mut entries: Vec<(String, String, u64)> = Vec::new();
    for line in content.lines() {
        if json_extract_str(line, "sid").as_deref() == Some(&sid) {
            let file = json_extract_str(line, "file").unwrap_or_default();
            let kind = json_extract_str(line, "type").unwrap_or_default();
            let ln = json_extract_u64(line, "line");
            entries.push((file, kind, ln));
        }
    }
    if entries.is_empty() {
        println!("fp manifest: no patches in current session.");
        return;
    }
    println!("fp Patch Manifest — current session ({} patches)", entries.len());
    println!("{}", "=".repeat(60));
    for (file, kind, ln) in &entries {
        let fname = Path::new(file).file_name()
            .and_then(|n| n.to_str()).unwrap_or(file);
        println!("  {} {:8} @ L{}", fname, kind, ln);
    }
}


// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = env::args().collect();
    match args.as_slice() {
        [_, flag] if flag == "--hook"       => hook_mode(),
        [_, flag] if flag == "--write-hook" => write_hook_mode(),
        [_, flag] if flag == "--read-hook"  => read_hook_mode(),
        [_, cmd]  if cmd  == "gain"         => gain_mode(),
        [_, cmd]  if cmd  == "manifest"     => manifest_mode(),
        [_, file, old_str, new_str] => {
            match apply_patch(file, old_str, new_str) {
                Ok(msg) => println!("{}", msg),
                Err(e)  => { eprintln!("{}", e); process::exit(1); }
            }
        }
        [_, flag, file, symbol, new_body] if flag == "--symbol" => {
            match apply_symbol_patch(file, symbol, new_body) {
                Ok(msg) => println!("{}", msg),
                Err(e)  => { eprintln!("{}", e); process::exit(1); }
            }
        }
        _ => {
            eprintln!("fp — token-efficient file patcher");
            eprintln!();
            eprintln!("Usage:");
            eprintln!("  fp <file> <old_str> <new_str>          search-replace patch");
            eprintln!("  fp --symbol <file> <symbol> <body>     replace symbol by name");
            eprintln!("  fp --hook                              Edit tool hook (stdin JSON)");
            eprintln!("  fp --write-hook                        Write tool hook (stdin JSON)");
            eprintln!("  fp gain                                show token savings");
    eprintln!("  fp manifest                            show this session\'s patches");
    eprintln!("  fp --read-hook                         Read tool hook (stdin JSON)");
            process::exit(2);
        }
    }
}
