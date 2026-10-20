#!/usr/bin/env bash
# Runtime: Bash 4+, awk and standard Linux utilities; no Python or jq.
# Model downloads use curl or wget. If both are absent, the interactive wizard
# can install curl through a supported system package manager after confirmation.
set -euo pipefail

fail() { printf '%s\n' "$*" >&2; return 1; }
need() { command -v "$1" >/dev/null || fail "Required system tool not found: $1"; }
trim() { REPLY=$1; REPLY="${REPLY#"${REPLY%%[![:space:]]*}"}"; REPLY="${REPLY%"${REPLY##*[![:space:]]}"}"; }
launcher_exit() {
    local status=$1 ignored
    trap - EXIT
    if ((status != 0)) && [[ $status != 130 && $status != 143 && ${INTERACTIVE-false} == true && -t 0 ]]; then
        printf '\nMoE4All launch failed (exit %d). Review the messages above.\nPress Enter to close: ' "$status" >&2
        IFS= read -r ignored || true
    fi
    exit "$status"
}

# JSON.awk 1.4.2 (MIT), embedded so the launcher needs no separate parser.
read -r -d '' JSON_PARSER <<'MOE4ALL_JSON_AWK' || true
# Vendored from step-/JSON.awk commit 7185bf0afc786ae3ca0250a3fda4dfcf3c23f0d3
# Local guards reject duplicate/escaped object keys and excessive nesting.
# The MIT License
#
# Copyright (c) 2013 step-
#
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to
# deal in the Software without restriction, including
# without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom
# the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice
# shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
# ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#!/usr/bin/awk -f
#
# Software: JSON.awk - a practical JSON parser written in awk
# Version: 1.4.2
# Copyright (c) 2013-2020, step
# License: MIT or Apache 2
# Project home: https://github.com/step-/JSON.awk
# Credits:      https://github.com/step-/JSON.awk#credits

# See README.md for full usage instructions.
# Usage:
#   awk [-v Option="value"...] -f JSON.awk "-" -or- Filepath [Filepath...]
#   printf "%s\n" Filepath [Filepath...] | awk [-v Option="value"...] -f JSON.awk
# Options: (default value in braces)
#    BRIEF=: 0 or M {1}:
#      non-zero excludes non-leaf nodes (array and object) from stdout; bit
#      mask M selects which to include of ""(1), "[]"(2) and "{}"(4), or
#      excludes ""(8) and wins over bit 1. BRIEF=0 includes all.
#   STREAM=: 0 or 1 {1}:
#      zero hooks callbacks into parser and stdout printing.
#   STRICT=: 0,1,2 {0}:
#      1 enforce RFC8259#7 character escapes except for solidus '/'
#      2 enforce solidus escape too (for JSON embedded in HTML/XML)

BEGIN { #{{{1
	if (BRIEF  == "") BRIEF=1  # when 1 parse() omits non-leaf nodes from stdout
	if (STREAM == "") STREAM=1 # when 0 parse() stores JPATHS[] for callback cb_jpaths
	if (STRICT == "") STRICT=1 # when 1 parse() enforces valid character escapes (RFC8259 7)

	# Set if empty string/array/object go to stdout and cb_jpaths when BRIEF>0
	# defaults compatible with version up to 1.2
	NO_EMPTY_STR = 0; NO_EMPTY_ARY = NO_EMPTY_OBJ = 1
	#  leaf             non-leaf       non-leaf

	if (BRIEF > 0) { # parse() will look at NO_EMPTY_*
		NO_EMPTY_STR = !(x=bit_on(BRIEF, 0))
		NO_EMPTY_ARY = !(x=bit_on(BRIEF, 1))
		NO_EMPTY_OBJ = !(x=bit_on(BRIEF, 2))
		if (x=bit_on(BRIEF, 3)) NO_EMPTY_STR = 1 # wins over bit 0
	}

	# for each input file:
	#   TOKENS[], NTOKENS, ITOKENS - tokens after tokenize()
	#   JPATHS[], NJPATHS - parsed data (when STREAM=0)
	# at script exit:
	#   FAILS[] - maps names of invalid files to logged error lines
	delete FAILS; delete OBJECT_KEYS
	reset()

	if (1 == ARGC) {
		# file pathnames from stdin
		# usage: echo -e "file1\nfile2\n" | awk -f JSON.awk
		# usage: { echo; cat file1; } | awk -f JSON.awk
		while (getline ARGV[++ARGC] < "/dev/stdin") {
			if (ARGV[ARGC] == "")
				break
		}
	} # else usage: awk -f JSON.awk file1 [file2...]

	# set file slurping mode
	srand(); RS="\1n/o/m/a/t/c/h" rand()
}

{ # main loop: process each file in turn {{{1
	reset() # See important application note in reset()

	++FILEINDEX # 1-based
	tokenize($0) # while(get_token()) {print TOKEN}
	if (0 == parse() && 0 == STREAM) {
		# Pass the callback an array of jpaths.
		cb_jpaths(JPATHS, NJPATHS)
	}
}

END { # process invalid files {{{1
	if (0 == STREAM) {
		# Pass the callback an associative array of failed objects.
		cb_fails(FAILS, NFAILS)
	}
	exit(NFAILS > 0)
}

function bit_on(n, b) { #{{{1
# Return n & (1 << b) for b>0 n>=0 - for awk portability
	if (b == 0) return n % 2
	return int(n / 2^b) % 2
}

function append_jpath_component(jpath, component) { #{{{1
	if (0 == STREAM) {
		return cb_append_jpath_component(jpath, component)
	} else {
		return (jpath != "" ? jpath "," : "") component
	}
}

function append_jpath_value(jpath, value) { #{{{1
	if (0 == STREAM) {
		return cb_append_jpath_value(jpath, value)
	} else {
		return sprintf("[%s]\t%s", jpath, value)
	}
}

function get_token() { #{{{1
# usage: {tokenize($0); while(get_token()) {print TOKEN}}

	# return getline TOKEN # for external tokenizer

	TOKEN = TOKENS[++ITOKENS] # for internal tokenize()
	return ITOKENS < NTOKENS  # 1 if more tokens to come
}

function parse_array_empty(jpath) { #{{{1
	if (0 == STREAM) {
		return cb_parse_array_empty(jpath)
	}
	return "[]"
}

function parse_array_enter(jpath) { #{{{1
	if (0 == STREAM) {
		cb_parse_array_enter(jpath)
	}
}

function parse_array_exit(jpath, status) { #{{{1
	if (0 == STREAM) {
		cb_parse_array_exit(jpath, status)
	}
}

function parse_array(a1,   idx,ary,ret) { #{{{1
	idx=0
	ary=""
	get_token()
#	print "parse_array(" a1 ") TOKEN=" TOKEN >"/dev/stderr"
	if (TOKEN != "]") {
		while (1) {
			if (ret = parse_value(a1, idx)) {
				return ret
			}
			idx=idx+1
			ary=ary VALUE
			get_token()
			if (TOKEN == "]") {
				break
			} else if (TOKEN == ",") {
				ary = ary ","
			} else {
				report(", or ]", TOKEN ? TOKEN : "EOF")
				return 2
			}
			get_token()
		}
		CB_VALUE = sprintf("[%s]", ary)
		# VALUE="" marks non-leaf jpath
		VALUE = 0 == BRIEF ? CB_VALUE : ""
	} else {
		VALUE = CB_VALUE = parse_array_empty(a1)
	}
	return 0
}

function parse_object_empty(jpath) { #{{{1
	if (0 == STREAM) {
		return cb_parse_object_empty(jpath)
	}
	return "{}"
}

function parse_object_enter(jpath) { #{{{1
	if (0 == STREAM) {
		cb_parse_object_enter(jpath)
	}
}

function parse_object_exit(jpath, status) { #{{{1
	if (0 == STREAM) {
		cb_parse_object_exit(jpath, status)
	}
}

function parse_object(a1,   key,obj) { #{{{1
	obj=""
	get_token()
#	print "parse_object(" a1 ") TOKEN=" TOKEN >"/dev/stderr"
	if (TOKEN != "}") {
		while (1) {
			if (TOKEN ~ /^".*"$/) {
				key=TOKEN
                if (index(key, "\\") || (a1 SUBSEP key) in OBJECT_KEYS) {
                    report("unique unescaped object key", TOKEN); return 3
                }
                OBJECT_KEYS[a1 SUBSEP key]=1
			} else {
				report("string", TOKEN ? TOKEN : "EOF")
				return 3
			}
			get_token()
			if (TOKEN != ":") {
				report(":", TOKEN ? TOKEN : "EOF")
				return 4
			}
			get_token()
			if (parse_value(a1, key)) {
				return 5
			}
			obj=obj key ":" VALUE
			get_token()
			if (TOKEN == "}") {
				break
			} else if (TOKEN == ",") {
				obj=obj ","
			} else {
				report(", or }", TOKEN ? TOKEN : "EOF")
				return 6
			}
			get_token()
		}
		CB_VALUE = sprintf("{%s}", obj)
		# VALUE="" marks non-leaf jpath
		VALUE = 0 == BRIEF ? CB_VALUE : ""
	} else {
		VALUE = CB_VALUE = parse_object_empty(a1)
	}
	return 0
}

function parse_value(a1, a2,   jpath,ret,x,reason) { #{{{1
	jpath = append_jpath_component(a1, a2)
    if (split(jpath, DEPTH_PARTS, ",") > 64) { report("bounded JSON depth", TOKEN); return 9 }
#	print "parse_value(" a1 "," a2 ") TOKEN=" TOKEN " jpath=" jpath >"/dev/stderr"

	if (TOKEN == "{") {
		parse_object_enter(jpath)
		if (parse_object(jpath)) {
			parse_object_exit(jpath, 7)
			return 7
		}
		parse_object_exit(jpath, 0)
	} else if (TOKEN == "[") {
		parse_array_enter(jpath)
		if (ret = parse_array(jpath)) {
			parse_array_exit(jpath, ret)
			return ret
		}
		parse_array_exit(jpath, 0)
	} else if (TOKEN == "") { #test case 20150410 #4
		report("value", "EOF")
		return 8
	} else if ((x = is_value(TOKEN)) >0) {
		CB_VALUE = VALUE = TOKEN
	} else {
		if (-1 == x || -2 == x) {
			reason = "missing or invalid character escape"
		}
		report("value", TOKEN, reason)
		return 9
	}

	# jpath=="" occurs on starting and ending the parsing session.
	# VALUE=="" is set on parsing a non-empty array or a non-empty object.
	# Either condition is a reason to discard the parsed jpath if BRIEF>0.
	if (0 < BRIEF && ("" == jpath || "" == VALUE)) {
		return 0
	}

	# BRIEF>1 is a bit mask that selects if an empty string/array/object is passed on
	if (0 < BRIEF && (NO_EMPTY_STR && VALUE=="\"\"" || NO_EMPTY_ARY && VALUE=="[]" || NO_EMPTY_OBJ && VALUE=="{}")) {
		return 0
	}

	x = append_jpath_value(jpath, VALUE)
	if(0 == STREAM) {
		# save jpath+value for cb_jpaths
		JPATHS[++NJPATHS] = x
	} else {
		# consume jpath+value directly
		print x
	}
	return 0
}

function parse(   ret) { #{{{1
	get_token()
	printf "__MOE4ALL_ROOT__\t%s\n", (TOKEN == "{" ? "object" : TOKEN == "[" ? "array" : "scalar")
	if (ret = parse_value()) {
		return ret
	}
	if (get_token() || "" != TOKEN) {
		report("EOF", TOKEN)
		return 10
		# TODO the next JSON text starts here.
	}
	return 0
}

function report(expected, got, extra,   i,from,to,context) { #{{{1
	from = ITOKENS - 10; if (from < 1) from = 1
	to = ITOKENS + 10; if (to > NTOKENS) to = NTOKENS
	for (i = from; i < ITOKENS; i++)
		context = context sprintf("%s ", TOKENS[i])
	context = context "<<" got ">> "
	for (i = ITOKENS + 1; i <= to; i++)
		context = context sprintf("%s ", TOKENS[i])
	scream("expected <" expected "> but got <" got "> (length " length(got) (extra ? ", "extra :"") ") at input token " ITOKENS "\n" context)
}

function reset() { #{{{1
# Application Note:
# If you need to build JPATHS[] incrementally from multiple input files:
# 1) Comment out below:        delete JPATHS; NJPATHS=0
#    otherwise each new input file would reset JPATHS[].
# 2) Move the call to apply() from the main loop to the END statement.
# 3) In the main loop consider adding code that deletes partial JPATHS[]
#    elements that would result from parsing invalid JSON files.
# Compatibility Note:
# 1) Very old gawk versions: replace 'delete JPATHS' with 'split("", JPATHS)'.

	TOKEN=""; delete TOKENS; NTOKENS=ITOKENS=0
	delete JPATHS; NJPATHS=0
	CB_VALUE = VALUE = ""
}

function scream(msg) { #{{{1
	NFAILS += (FILENAME in FAILS ? 0 : 1)
	FAILS[FILENAME] = FAILS[FILENAME] (FAILS[FILENAME]!="" ? "\n" : "") msg
	if(0 == STREAM) {
		if(cb_fail1(msg)) {
			print FILENAME ": " msg >"/dev/stderr"
		}
	} else {
		print FILENAME ": " msg >"/dev/stderr"
	}
}

function tokenize(a1) { #{{{1
# usage A: {for(i=1; i<=tokenize($0); i++) print TOKENS[i]}
# see also get_token()

# Pattern string summary with adjustments:
# - replace strings with regex constant; https://github.com/step-/JSON.awk/issues/1
# - reduce [:cntrl:] to [\000-\037]; https://github.com/step-/JSON.awk/issues/5
# - reduce [:space:] to [ \t\n\r]; https://tools.ietf.org/html/rfc8259#page-5 ws
# - replace {4} quantifier with three [0-9a-fA-F] for mawk; https://unix.stackexchange.com/a/506125
# - UTF-8 BOM signature; https://en.wikipedia.org/wiki/Byte_order_mark#Byte_order_marks_by_encoding
# ----------
# 	TOKENS  = BOM "|" STRING "|" NUMBER "|" KEYWORD "|" SPACE "|."
# 	BOM     = "^\357\273\277"  # cf. issue #17
# 	STRING  = "\"" CHAR "*(" ESCAPE CHAR "*)*\""
# 	ESCAPE  = "(\\[^u[:cntrl:]]|\\u[0-9a-fA-F]{4})"
# 	CHAR    = "[^[:cntrl:]\\\"]"
# 	NUMBER  = "-?(0|[1-9][0-9]*)([.][0-9]+)?([eE][+-]?[0-9]+)?"
# 	KEYWORD = "null|false|true"
# 	SPACE   = "[[:space:]]+"

	gsub(/^\357\273\277|"[^"\\\000-\037]*((\\[^u\000-\037]|\\u[0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F])[^"\\\000-\037]*)*"|-?(0|[1-9][0-9]*)([.][0-9]+)?([eE][+-]?[0-9]+)?|null|false|true|[ \t\n\r]+|./, "\n&", a1)
	gsub("\n" "[ \t\n\r]+", "\n", a1)
	# ^\n BOM or \n$?
	gsub(/^\n(\357\273\277\n)?|\n$/, "", a1)
	ITOKENS=0 # get_token() helper
	return NTOKENS = split(a1, TOKENS, /\n/)
}

function is_value(a1) { #{{{1
	# Return 0(malformed <value>) <0(<value> but !strict content) >0(pass)

	# STRING | NUMBER | KEYWORD
	if(!STRICT)
		return a1 ~ /^("[^"\\\000-\037]*((\\[^u\000-\037]|\\u[0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F])[^"\\\000-\037]*)*"|-?(0|[1-9][0-9]*)([.][0-9]+)?([eE][+-]?[0-9]+)?|null|false|true)$/

	# STRICT is on
	# unescaped = %x20-21 / %x23-5B / %x5D-10FFFF
	# Characters in a STRING are restricted as follows (RFC8259):
	# All Unicode characters may be placed within the quotation marks, except for the characters that MUST be escaped:
	# quotation mark, reverse solidus, and the control characters (U+0000 through U+001F).
	# Any character may be escaped with \uXXXX, alternatively, with the following two-character escapes:
	# %x75 4HEXDIG    ; uXXXX                U+XXXX
	# %x22 /          ; "    quotation mark  U+0022
	# %x5C /          ; \    reverse solidus U+005C
	# %x62 /          ; b    backspace       U+0008
	# %x66 /          ; f    form feed       U+000C
	# %x6E /          ; n    line feed       U+000A   removed by tokenizer
	# %x72 /          ; r    carriage return U+000D   removed by tokenizer
	# %x2F /          ; /    solidus         U+002F   enforced only when STRICT >1
	# %x74 /          ; t    tab             U+0009   removed by tokenizer

	# NUMBER | KEYWORD
	if (1 != index(a1, "\"")) {
		return a1 ~ /^(-?(0|[1-9][0-9]*)([.][0-9]+)?([eE][+-]?[0-9]+)?|null|false|true)$/
	}
	# invalid STRING
	if (a1 !~ /^("[^"\\\000-\037]*((\\[^u\000-\037]|\\u[0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F])[^"\\\000-\037]*)*")$/) {
		return 0
	}
	a1 = substr(a1, 2, length(a1) -2)

	# STRICT 1: allowed character escapes
	gsub(/\\["\\\/bfnrt]|\\u[0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F]/, "", a1)
	# STRICT 1: unescaped quotation-mark, reverse solidus and control characters
	if (a1 ~ /["\\\000-\037]/) {
		return -1
	}
	# STRICT 2: unescaped solidus
	if (STRICT > 1 && index(a1, "/")) {
		return -2
	}
	# PASS STRICT STRING
	return 1
}

# The launcher always uses STREAM=1, so these JSON.awk extension hooks are never
# called. mawk still resolves every referenced function while compiling the
# program, unlike gawk, and rejects the parser when the optional hooks are absent.
function cb_jpaths(a, n, k) { for (k in a) break }
function cb_fails(a, n, k) { for (k in a) break }
function cb_append_jpath_component(a, b) { return "" }
function cb_append_jpath_value(a, b) { return "" }
function cb_parse_array_empty(a) { return "[]" }
function cb_parse_array_enter(a) {}
function cb_parse_array_exit(a, b) {}
function cb_parse_object_empty(a) { return "{}" }
function cb_parse_object_enter(a) {}
function cb_parse_object_exit(a, b) {}
function cb_fail1(a) { return 0 }

# vim:fdm=marker:
MOE4ALL_JSON_AWK

declare -A S=() DEFAULTS=() J=() OVERRIDES=() BENCHMARK_ESTIMATE_MINUTES=()
BENCHMARK_ESTIMATE_PID=''
BENCHMARK_ESTIMATE_FD=''
BENCHMARK_ESTIMATE_ERROR=''
declare -a COMMAND=() FILES=() CANDIDATES=()

json_load() {
    local flat path value
    J=(); JSON_KIND=''
    [[ -s $1 && $(wc -c < "$1") -le 8388608 ]] || return 1
    flat=$(LC_ALL=C awk "$JSON_PARSER" "$1" 2>/dev/null) || return 1
    while IFS=$'\t' read -r path value; do
        if [[ $path == __MOE4ALL_ROOT__ ]]; then JSON_KIND=$value; continue; fi
        [[ -n $path && ! -v 'J[$path]' ]] || return 1
        J["$path"]=$value
    done <<< "$flat"
    [[ $JSON_KIND == object || $JSON_KIND == array ]]
}
utf8_char() {
    local n=$1 byte encoded='' octal
    local -a bytes=()
    ((n>0 && n<=1114111 && (n<55296 || n>57343))) || return 1
    if ((n<128)); then bytes=("$n")
    elif ((n<2048)); then bytes=($((192+n/64)) $((128+n%64)))
    elif ((n<65536)); then bytes=($((224+n/4096)) $((128+n/64%64)) $((128+n%64)))
    else bytes=($((240+n/262144)) $((128+n/4096%64)) $((128+n/64%64)) $((128+n%64))); fi
    for byte in "${bytes[@]}"; do printf -v octal '%03o' "$byte"; encoded+="\\$octal"; done
    printf -v REPLY '%b' "$encoded"
}
json_decode() {
    local text=$1 out='' c hex n low i
    [[ $text == \"*\" ]] || { REPLY=$text; return; }
    text=${text:1:${#text}-2}
    for ((i=0;i<${#text};i++)); do
        c=${text:i:1}
        if [[ $c != '\' ]]; then out+=$c; continue; fi
        ((i+=1)); c=${text:i:1}
        case $c in
            '"'|'\'|/) out+=$c ;; b) out+=$'\b' ;; f) out+=$'\f' ;; n) out+=$'\n' ;; r) out+=$'\r' ;; t) out+=$'\t' ;;
            u)
                hex=${text:i+1:4}; [[ $hex =~ ^[0-9a-fA-F]{4}$ ]] || return 1
                n=$((16#$hex)); ((i+=4))
                if ((n>=55296 && n<=56319)); then
                    [[ ${text:i+1:2} == '\u' ]] || return 1
                    hex=${text:i+3:4}; [[ $hex =~ ^[0-9a-fA-F]{4}$ ]] || return 1
                    low=$((16#$hex)); ((low>=56320 && low<=57343)) || return 1
                    n=$((65536+(n-55296)*1024+low-56320)); ((i+=6))
                fi
                utf8_char "$n" || return 1; out+=$REPLY ;;
            *) return 1 ;;
        esac
    done
    REPLY=$out
}
json_get() { local path=$1; [[ -v 'J[$path]' ]] || return 1; json_decode "${J[$path]}"; }
json_quote() {
    local text=$1 out='"' c escaped i byte LC_ALL=C
    for ((i=0;i<${#text};i++)); do
        c=${text:i:1}
        case $c in
            '"') out+='\"' ;; '\') out+='\\' ;; $'\n') out+='\n' ;; $'\r') out+='\r' ;; $'\t') out+='\t' ;;
            *) printf -v byte '%d' "'$c"; if ((byte<32)); then printf -v escaped '\\u%04x' "$byte"; out+=$escaped; else out+=$c; fi ;;
        esac
    done
    REPLY="$out\""
}
defaults() {
    local key
    S=(
        [launch_mode]=run [setup_mode]=conservative [model]='' [model_history]='' [device]='' [context]=''
        [ubatch]='' [threads]='' [config_path]='' [kv_preset]=q8 [kv_type_k]=q8_0 [kv_type_v]=q8_0
        [configure_memory]=false [vram_budget]='' [vram_reserve]='' [expert_cache]='' [ram_budget]=''
        [host_dma]=true [pager_ring]='' [pager_ring_slots]='' [kv_overflow]=false [kv_overflow_vram_mb]='' [kv_overflow_reserve_mb]=''
        [submit_mode]=auto [submit_cap]=64 [configure_diagnostics]=false [pager_stats]=false [pager_profile]=false [stage_profile]=false [vram_profile]=false
        [think_mode]=default [configure_thinking]=false [reasoning_effort]=default [max_new]=65536
        [configure_sampling]=false [temperature]='' [top_k]='' [top_p]='' [seed]=''
        [server_addr]=127.0.0.1:8080 [server_parallel]=1 [server_auth]=false [server_session_cache]=true
        [session_cache_dir]="$ROOT/kv-sessions" [session_idle_secs]=120 [session_cache_max]=5GiB [session_cache_ttl_hours]=24
        [server_vision]=false [vision_projector]='' [server_embedding]=false [embedding_model]='' [embedding_idle_timeout]=300
        [mtp_enabled]=false [mtp_model]='' [mtp_verify_tokens]=4 [cpu_miss_enabled]=false [cpu_miss_max]=1 [cpu_miss_cores]=''
        [bench_kind]=quick [bench_parallel]=1 [manual_bench_kind]=decode [prompt_tokens]=6144 [gen_tokens]=512 [depth_mode]=none [depth_tokens]=0 [reps]=1 [json_output]=false
        [custom_sets]='' [auto_overrides]='' [benchmark_upload]=false [benchmark_upload_has_github]=true [benchmark_upload_gh_auth]=true [error_upload]=false [submission_browser_action]=later [submission_failure_action]=later [submission_pending_retry]=true
    )
    DEFAULTS=(); for key in "${!S[@]}"; do DEFAULTS[$key]=${S[$key]}; done
}
load_state() {
    local key raw value path source_state='' source_conf=''
    if [[ -f $STATE ]]; then
        source_state=$STATE
    elif [[ -n ${LEGACY_STATE-} && -f $LEGACY_STATE ]]; then
        source_state=$LEGACY_STATE
    fi
    if [[ -n $source_state ]]; then
        json_load "$source_state" || { fail "Invalid saved settings: $source_state"; return 1; }
        [[ $JSON_KIND == object ]] || { fail 'Saved settings must be a JSON object.'; return 1; }
        for key in "${!S[@]}"; do
            path="[\"$key\"]"; [[ -v 'J[$path]' ]] || continue; raw=${J[$path]}
            case ${DEFAULTS[$key]} in
                true|false) [[ $raw == true || $raw == false ]] || { fail "Invalid saved boolean: $key"; return 1; } ;;
                *) [[ $raw == \"*\" ]] || { fail "Invalid saved string: $key"; return 1; } ;;
            esac
            json_decode "$raw" || return 1; S[$key]=$REPLY
        done
        path='["configure_thinking"]'
        if [[ ! -v 'J[$path]' && (${S[think_mode]} != default || ${S[reasoning_effort]} != default) ]]; then S[configure_thinking]=true; fi
    else
        if [[ -f $CONFIG_DIR/wizard.conf ]]; then
            source_conf=$CONFIG_DIR/wizard.conf
        elif [[ -n ${LEGACY_CONFIG_DIR-} && -f $LEGACY_CONFIG_DIR/wizard.conf ]]; then
            source_conf=$LEGACY_CONFIG_DIR/wizard.conf
        fi
    fi
    if [[ -n $source_conf ]]; then
        local -A names=([MODE]=launch_mode [MODEL]=model [PROFILE]=setup_mode [CTX]=context [UBATCH]=ubatch [KV_K]=kv_type_k [KV_V]=kv_type_v
            [RAM]=ram_budget [VRAM]=vram_budget [MTP]=mtp_model [MTP_K]=mtp_verify_tokens [ADDR]=server_addr [PARALLEL]=server_parallel [MMPROJ]=vision_projector [EMBEDDING]=embedding_model)
        while IFS= read -r value; do
            [[ $value == *=* ]] || continue; key=${value%%=*}; [[ ! -v 'names[$key]' ]] || S[${names[$key]}]=${value#*=}
        done < "$source_conf"
        [[ -z ${S[mtp_model]} ]] || S[mtp_enabled]=true
        [[ -z ${S[vision_projector]} ]] || S[server_vision]=true
        [[ -z ${S[embedding_model]} ]] || S[server_embedding]=true
        [[ -z ${S[ram_budget]}${S[vram_budget]} ]] || S[configure_memory]=true
        for key in ubatch ram_budget vram_budget; do [[ -z ${S[$key]} ]] || S[auto_overrides]+="$key,"; done
    fi
    case ${S[launch_mode]} in chat) S[launch_mode]=run ;; server) S[launch_mode]=serve ;; benchmark) S[launch_mode]=bench ;; esac
}
save_state() {
    local temporary key sep=''
    mkdir -p -- "$CONFIG_DIR"; temporary=$(mktemp "$CONFIG_DIR/.wizard-state.XXXXXX")
    {
        printf '{\n'
        for key in "${!S[@]}"; do
            printf '%s  "%s": ' "$sep" "$key"
            case ${DEFAULTS[$key]} in true|false) printf '%s' "${S[$key]}" ;; *) json_quote "${S[$key]}"; printf '%s' "$REPLY" ;; esac
            sep=$',\n'
        done
        printf '\n}\n'
    } > "$temporary"
    chmod 600 "$temporary"; mv -f -- "$temporary" "$STATE"
}
prompt() {
    local value
    while true; do
        printf '%s' "$1"; [[ -z ${2-} ]] || printf ' [%s]' "$2"; printf ': '
        IFS= read -r value || { fail '启动已取消：输入已结束。Launch cancelled: input ended.'; return 1; }; trim "$value"; value=$REPLY
        if [[ $value == - ]]; then value=''; elif [[ -z $value ]]; then value=${2-}; fi
        if [[ -n $value || ${3-false} == false ]]; then REPLY=$value; return; fi
        printf '此项不能为空。A value is required.\n'
    done
}
ask() { prompt "$2" "${S[$1]}" "${3-false}"; S[$1]=$REPLY; }
yes() {
    local value
    while true; do
        if [[ ${2-false} == true ]]; then printf '%s [Y/n]: ' "$1"; else printf '%s [y/N]: ' "$1"; fi
        IFS= read -r value || { fail '启动已取消：输入已结束。Launch cancelled: input ended.'; return 1; }; trim "$value"; value=${REPLY,,}
        case $value in '') REPLY=${2-false}; return ;; y|yes) REPLY=true; return ;; n|no) REPLY=false; return ;; esac
        printf '请输入 y 或 n。Enter y or n.\n'
    done
}
ask_yes() { yes "$2" "${S[$1]}"; S[$1]=$REPLY; }
valid_uint() { [[ $1 =~ ^[0-9]{1,9}$ ]] && ((10#$1 >= ${2-0} && 10#$1 <= ${3-999999999})); }
integer() {
    while true; do
        prompt "$2" "${S[$1]}"
        if [[ -z $REPLY && ${5-false} == true ]] || valid_uint "$REPLY" "${3-0}" "${4-999999999}"; then S[$1]=$REPLY; return; fi
        printf '整数超出允许范围。Integer outside the allowed range.\n'
    done
}
choice() {
    local key=$1 label=$2 selected i entry; shift 2
    local rest detail; local -a values=() labels=() details=()
    for entry in "$@"; do
        values+=("${entry%%|*}"); rest=${entry#*|}
        if [[ $rest == *'|'* ]]; then labels+=("${rest%%|*}"); details+=("${rest#*|}")
        else labels+=("$rest"); details+=(''); fi
    done
    printf '\n%s\n' "$label"
    for ((i=0;i<${#values[@]};i++)); do
        selected=' '; [[ ${S[$key]} != "${values[i]}" ]] || selected='*'; printf ' %s[%d] %s\n' "$selected" "$((i+1))" "${labels[i]}"
        if [[ -n ${details[i]} ]]; then
            while IFS= read -r detail; do
                if [[ -t 1 ]]; then printf '\033[90m       %s\033[0m\n' "$detail"; else printf '       %s\n' "$detail"; fi
            done <<< "${details[i]}"
        fi
    done
    while true; do
        printf '选择（回车复用上次值/默认值）/ Select (Enter = previous/default): '
        IFS= read -r selected || { fail '启动已取消：输入已结束。Launch cancelled: input ended.'; return 1; }; trim "$selected"; selected=$REPLY
        if [[ -z $selected ]]; then
            for entry in "${values[@]}"; do [[ ${S[$key]} != "$entry" ]] || return 0; done
        elif valid_uint "$selected" 1 "${#values[@]}"; then
            S[$key]=${values[10#$selected-1]}; return 0
        fi
        for ((i=0;i<${#values[@]};i++)); do
            if [[ $selected == "${values[i]}" ]]; then S[$key]=${values[i]}; return; fi
        done
        printf '请选择列表中的选项。Choose a listed option.\n'
    done
}

benchmark_estimate_args() {
    ESTIMATE_ARGS=("$BINARY" bench --estimate-only)
    if [[ -n ${S[device]} ]]; then ESTIMATE_ARGS+=(--dev "${S[device]}"); fi
    if [[ ${S[setup_mode]} == manual ]]; then
        [[ -z ${S[config_path]} ]] || ESTIMATE_ARGS+=(--config "${S[config_path]}")
        [[ -z ${S[ubatch]} ]] || ESTIMATE_ARGS+=(--ubatch "${S[ubatch]}")
        [[ -z ${S[threads]} ]] || ESTIMATE_ARGS+=(--threads "${S[threads]}")
        if [[ ${S[configure_memory]-false} == true ]]; then
            [[ -z ${S[vram_budget]} ]] || ESTIMATE_ARGS+=(--set "device.vram_budget=${S[vram_budget]}")
            [[ -z ${S[vram_reserve]} ]] || ESTIMATE_ARGS+=(--set "device.vram_reserve=${S[vram_reserve]}")
            [[ -z ${S[ram_budget]} ]] || ESTIMATE_ARGS+=(--set "device.ram_budget=${S[ram_budget]}")
        fi
    else
        ESTIMATE_ARGS+=(--set "device.auto_profile=${S[setup_mode]}")
    fi
    ESTIMATE_ARGS+=("${S[model]}")
}

start_benchmark_estimate_probe() {
    [[ ${S[launch_mode]} == bench && -z $BENCHMARK_ESTIMATE_PID && -z $BENCHMARK_ESTIMATE_ERROR && ${#BENCHMARK_ESTIMATE_MINUTES[@]} -eq 0 ]] || return 0
    benchmark_estimate_args
    coproc MOE4ALL_BENCH_ESTIMATE { "${ESTIMATE_ARGS[@]}" 2>/dev/null; }
    BENCHMARK_ESTIMATE_PID=$MOE4ALL_BENCH_ESTIMATE_PID
    BENCHMARK_ESTIMATE_FD=${MOE4ALL_BENCH_ESTIMATE[0]}
}

complete_benchmark_estimate_probe() {
    local output line status suite parallel index value
    local -a fields=()
    ((${#BENCHMARK_ESTIMATE_MINUTES[@]} == 0)) || return 0
    [[ -z $BENCHMARK_ESTIMATE_ERROR ]] || return 0
    [[ -n $BENCHMARK_ESTIMATE_PID ]] || start_benchmark_estimate_probe
    [[ -n $BENCHMARK_ESTIMATE_PID ]] || return 0
    output=$(cat <&"$BENCHMARK_ESTIMATE_FD")
    if wait "$BENCHMARK_ESTIMATE_PID"; then status=0; else status=$?; fi
    BENCHMARK_ESTIMATE_PID=''; BENCHMARK_ESTIMATE_FD=''
    if ((status != 0)); then BENCHMARK_ESTIMATE_ERROR="estimate process exited with code $status"; return 0; fi
    line=$(printf '%s\n' "$output" | awk '/^BENCH_ESTIMATE_MENU_V1\|/{last=$0} END{print last}')
    IFS='|' read -r -a fields <<< "$line"
    if ((${#fields[@]} != 13)) || [[ ${fields[0]-} != BENCH_ESTIMATE_MENU_V1 ]]; then
        BENCHMARK_ESTIMATE_ERROR='estimate matrix protocol response is missing'; return 0
    fi
    index=1
    for suite in quick standard full; do
        for parallel in 1 2 3 4; do
            value=${fields[index]}
            if [[ ! $value =~ ^[1-9][0-9]*$ ]]; then
                BENCHMARK_ESTIMATE_MINUTES=(); BENCHMARK_ESTIMATE_ERROR='estimate matrix contains invalid minutes'; return 0
            fi
            BENCHMARK_ESTIMATE_MINUTES["$suite:$parallel"]=$value
            ((index+=1))
        done
    done
}

benchmark_estimate_detail() {
    local suite=$1 parallel=$2 key
    complete_benchmark_estimate_probe
    key="$suite:$parallel"
    if [[ -n ${BENCHMARK_ESTIMATE_MINUTES[$key]-} ]]; then
        REPLY="预计约 ${BENCHMARK_ESTIMATE_MINUTES[$key]} 分钟。Estimated about ${BENCHMARK_ESTIMATE_MINUTES[$key]} min."
    else
        REPLY='暂时无法估算。Estimate currently unavailable.'
    fi
}

absolute() {
    local value=$1
    [[ $value != '~' ]] || value=$HOME
    [[ $value != '~/'* ]] || value="$HOME/${value:2}"
    [[ $value == /* ]] || value="$ROOT/$value"
    REPLY=$(realpath -m -- "$value")
}
CATALOG_IDS=(qwen35-balanced flashnext-427 flashnext-vision qwen3-embedding-06b-q8)
catalog_load() {
    CATALOG_ID=$1 CATALOG_KIND='' CATALOG_NAME='' CATALOG_INFO='' CATALOG_PRIMARY='' CATALOG_COUNT=0
    case $1 in
        flashnext-427)
            CATALOG_KIND=llm; CATALOG_NAME='Qwen3.8 Flash-Next AD-4.27bpw-Q4_K_M-M64 (33 shards)'
            CATALOG_INFO='https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64'
            CATALOG_PRIMARY='Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf'; CATALOG_COUNT=33 ;;
        qwen35-balanced)
            CATALOG_KIND=llm; CATALOG_NAME='Qwen3.6 35B APEX-I-Balanced'
            CATALOG_INFO='https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-GGUF/blob/main/Qwen3.6-35B-A3B-APEX-I-Balanced.gguf'
            CATALOG_PRIMARY='Qwen3.6-35B-A3B-APEX-I-Balanced/Qwen3.6-35B-A3B-APEX-I-Balanced.gguf'; CATALOG_COUNT=1 ;;
        flashnext-vision)
            CATALOG_KIND=vision; CATALOG_NAME='Qwen3.8 Flash-Next F16 vision projector'
            CATALOG_INFO='https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/blob/main/mmproj-Qwen3.8-Flash-Next-F16.gguf'
            CATALOG_PRIMARY='Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64/mmproj-Qwen3.8-Flash-Next-F16.gguf'; CATALOG_COUNT=1 ;;
        qwen3-embedding-06b-q8)
            CATALOG_KIND=embedding; CATALOG_NAME='Qwen3-Embedding 0.6B Q8_0'
            CATALOG_INFO='https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/blob/main/Qwen3-Embedding-0.6B-Q8_0.gguf'
            CATALOG_PRIMARY='Qwen3-Embedding-0.6B-Q8_0/Qwen3-Embedding-0.6B-Q8_0.gguf'; CATALOG_COUNT=1 ;;
        *) fail "Unknown recommended download: $1"; return 1 ;;
    esac
}
catalog_file() {
    local index=$2 file directory='Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64'
    catalog_load "$1" || return 1
    valid_uint "$index" 1 "$CATALOG_COUNT" || { fail "Invalid catalog file index: $index"; return 1; }
    case $1 in
        flashnext-427)
            printf -v file 'Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-%05d-of-00033.gguf' "$index"
            DOWNLOAD_RELATIVE="$directory/$file"
            DOWNLOAD_URL="https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/resolve/main/$directory/$file?download=true" ;;
        qwen35-balanced)
            DOWNLOAD_RELATIVE=$CATALOG_PRIMARY
            DOWNLOAD_URL='https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-GGUF/resolve/main/Qwen3.6-35B-A3B-APEX-I-Balanced.gguf?download=true' ;;
        flashnext-vision)
            DOWNLOAD_RELATIVE=$CATALOG_PRIMARY
            DOWNLOAD_URL='https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/resolve/main/mmproj-Qwen3.8-Flash-Next-F16.gguf?download=true' ;;
        qwen3-embedding-06b-q8)
            DOWNLOAD_RELATIVE=$CATALOG_PRIMARY
            DOWNLOAD_URL='https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf?download=true' ;;
    esac
}
recommendations() {
    local kind=${1-all} id
    printf '\n官方推荐的模型与组件 / Official recommended models and components:\n'
    for id in "${CATALOG_IDS[@]}"; do
        catalog_load "$id"
        [[ $kind == all || $CATALOG_KIND == "$kind" ]] || continue
        printf ' %s\n   %s\n' "$CATALOG_NAME" "$CATALOG_INFO"
    done
    printf '在模型选择中按 D，可将一个或多个项目下载到 %s/models。\nChoose D in a model prompt to download one or more items into %s/models.\n' "$ROOT" "$ROOT"
}
catalog_candidates() {
    local kind=$1 id path found existing
    for id in "${CATALOG_IDS[@]}"; do
        catalog_load "$id"; [[ $CATALOG_KIND == "$kind" ]] || continue
        path="$ROOT/models/$CATALOG_PRIMARY"; [[ -f $path ]] || continue
        found=false; for existing in "${CANDIDATES[@]}"; do [[ $existing != "$path" ]] || found=true; done
        [[ $found == true ]] || CANDIDATES+=("$path")
    done
}
run_privileged() {
    if ((EUID==0)); then "$@"
    elif command -v sudo >/dev/null 2>&1; then sudo "$@"
    else fail '安装 curl 需要 root 或 sudo。Installing curl requires root or sudo.'; return 1
    fi
}
install_curl() {
    if command -v apt-get >/dev/null 2>&1; then run_privileged apt-get update && run_privileged apt-get install -y curl
    elif command -v dnf >/dev/null 2>&1; then run_privileged dnf install -y curl
    elif command -v yum >/dev/null 2>&1; then run_privileged yum install -y curl
    elif command -v zypper >/dev/null 2>&1; then run_privileged zypper --non-interactive install curl
    elif command -v pacman >/dev/null 2>&1; then run_privileged pacman -Sy --noconfirm curl
    elif command -v apk >/dev/null 2>&1; then run_privileged apk add curl
    else fail '未找到 curl、wget 或受支持的包管理器。请安装 curl 或 wget 后重试。Neither curl/wget nor a supported package manager was found.'; return 1
    fi
    command -v curl >/dev/null 2>&1 || { fail 'curl 安装完成后仍不可用。curl installation completed but curl is still unavailable.'; return 1; }
}
ensure_model_downloader() {
    if command -v curl >/dev/null 2>&1; then MODEL_DOWNLOADER=curl; return 0; fi
    if command -v wget >/dev/null 2>&1; then MODEL_DOWNLOADER=wget; return 0; fi
    [[ ${INTERACTIVE-false} == true && ${HEADLESS-false} == false ]] || {
        fail '下载模型需要 curl 或 wget。请安装其中之一，或交互运行向导以安装 curl。Model download requires curl or wget.'; return 1;
    }
    yes '缺少 curl 和 wget。现在使用系统包管理器安装 curl？/ curl and wget are missing. Install curl now?' false
    [[ $REPLY == true ]] || return 1
    install_curl || return 1
    MODEL_DOWNLOADER=curl
}
model_download_file() {
    local url=$1 destination=$2 label=$3 partial attempt
    [[ $url =~ ^https://huggingface\.co/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+/resolve/[A-Za-z0-9._/?=-]+$ ]] || {
        fail "Refusing non-catalog download URL: $url"; return 1;
    }
    if [[ -s $destination ]]; then printf '已存在 / Already present: %s\n' "$destination"; return 0; fi
    [[ ! -e $destination ]] || rm -f -- "$destination"
    mkdir -p -- "${destination%/*}"; partial="$destination.part"
    ensure_model_downloader || return 1
    printf '\n%s\n' "$label"
    for attempt in 1 2 3; do
        if [[ $MODEL_DOWNLOADER == curl ]]; then
            if curl --fail --location --proto '=https' --proto-redir '=https' --connect-timeout 30 \
                --continue-at - --output "$partial" --progress-bar --user-agent 'MoE4All-Linux-Wizard/1.0' "$url"; then
                [[ -s $partial ]] || { fail 'Downloaded file is empty.'; return 1; }
                mv -f -- "$partial" "$destination"; return 0
            fi
        else
            local -a wget_args=(-c -O "$partial")
            if wget --help 2>&1 | grep -q -- '--https-only'; then wget_args+=(--https-only); fi
            if wget "${wget_args[@]}" "$url"; then
                [[ -s $partial ]] || { fail 'Downloaded file is empty.'; return 1; }
                mv -f -- "$partial" "$destination"; return 0
            fi
        fi
        printf '下载中断，正在从 .part 续传（%d/3）。Download interrupted; retrying from .part.\n' "$attempt" >&2
        ((attempt==3)) || sleep $((attempt*2))
    done
    fail '下载重试 3 次后仍失败。Download failed after 3 attempts.'
}
download_bundle() {
    local id=$1 i destination
    catalog_load "$id" || return 1
    printf '\n下载 / Download: %s\n目录 / Folder: %s/models\n' "$CATALOG_NAME" "$ROOT"
    for ((i=1;i<=CATALOG_COUNT;i++)); do
        catalog_file "$id" "$i"
        destination="$ROOT/models/$DOWNLOAD_RELATIVE"
        model_download_file "$DOWNLOAD_URL" "$destination" "[$i/$CATALOG_COUNT] ${destination##*/}" || return 1
    done
    catalog_load "$id"
    [[ -s $ROOT/models/$CATALOG_PRIMARY ]] || { fail "Downloaded primary file is missing: $ROOT/models/$CATALOG_PRIMARY"; return 1; }
    printf '下载完成 / Download complete: %s/models/%s\n' "$ROOT" "$CATALOG_PRIMARY"
}
download_menu() {
    local raw token number index id primary i valid; local -a selected=(); local -A seen=()
    while true; do
        printf '\n下载模型与组件 / Download models and components\n'
        for ((i=0;i<${#CATALOG_IDS[@]};i++)); do
            catalog_load "${CATALOG_IDS[i]}"; primary="$ROOT/models/$CATALOG_PRIMARY"
            if [[ -s $primary ]]; then printf ' [%d] %s [已安装 / installed]\n' "$((i+1))" "$CATALOG_NAME"; else printf ' [%d] %s\n' "$((i+1))" "$CATALOG_NAME"; fi
            printf '     %s\n' "$CATALOG_INFO"
        done
        printf ' [A] 全部 / All\n [B] 返回 / Back\n选择一个或多个编号，用逗号分隔 / Select one or more numbers, comma-separated: '
        IFS= read -r raw || { fail '下载选择已取消：输入已结束。Download selection cancelled: input ended.'; return 1; }; trim "$raw"; raw=$REPLY
        case ${raw,,} in b|back) return 0 ;; a|all) selected=(0 1 2 3) ;; *)
            raw=${raw//,/ }; raw=${raw//;/ }; selected=(); seen=(); valid=true
            for token in $raw; do
                if ! valid_uint "$token" 1 "${#CATALOG_IDS[@]}"; then valid=false; break; fi
                number=$((10#$token)); index=$((number-1))
                [[ -v 'seen[$index]' ]] || { selected+=("$index"); seen[$index]=1; }
            done
            if [[ $valid != true || ${#selected[@]} == 0 ]]; then printf '请选择列表中的编号，并用逗号分隔。Choose listed numbers separated by commas.\n'; continue; fi ;;
        esac
        yes '现在开始或继续所选下载？/ Start or resume the selected downloads now?' true; [[ $REPLY == true ]] || return 0
        for index in "${selected[@]}"; do id=${CATALOG_IDS[index]}; download_bundle "$id" || return 1; done
        return 0
    done
}
model_candidates() {
    local file lower
    CANDIDATES=(); [[ -d $1 ]] || return 0
    while IFS= read -r -d '' file; do
        lower=${file##*/}; lower=${lower,,}; [[ $lower == *.gguf ]] || continue
        case $2 in
            vision) [[ $lower == mmproj* ]] || continue ;;
            llm|mtp)
                [[ $lower != mmproj* ]] || continue
                if [[ $lower =~ -[0-9]{5}-of-[0-9]{5}\.gguf$ && ! $lower =~ -00001-of-[0-9]{5}\.gguf$ ]]; then continue; fi
                [[ $2 != llm || $lower != mtp* ]] || continue ;;
        esac
        CANDIDATES+=("$file")
    done < <(find "$1" -maxdepth 1 -type f -print0 | sort -z)
}
hub_reference() { [[ $1 != /* && ${1,,} != *.gguf && $1 == */* ]]; }
model_path() {
    local kind=$1 key=$2 value i found saved=${S[$2]} path
    model_candidates "$ROOT" "$kind"
    catalog_candidates "$kind"
    local -a nearby=("${CANDIDATES[@]}") remembered=("$saved")
    if [[ $kind == llm && -n ${S[model_history]} ]]; then
        while IFS= read -r value; do [[ -z $value ]] || remembered+=("$value"); done <<< "${S[model_history]}"
    fi
    if [[ $kind == llm && -f $ROOT/gui-data/state.json ]]; then
        if json_load "$ROOT/gui-data/state.json"; then
            for path in "${!J[@]}"; do
                if [[ $path =~ ^\[\"(recent|favorites)\",[0-9]+\]$ || $path =~ ^\[\"profiles\",[0-9]+,\"model_path\"\]$ ]]; then
                    json_get "$path" && remembered+=("$REPLY")
                fi
            done
        fi
    fi
    for value in "${remembered[@]}"; do
        [[ -n $value ]] || continue; absolute "$value"; value=$REPLY
        [[ -f $value && ${value,,} == *.gguf ]] || continue
        found=false; for path in "${nearby[@]}"; do [[ $path != "$value" ]] || found=true; done
        [[ $found == true ]] || nearby=("$value" "${nearby[@]}")
    done
    while true; do
        printf '\n%s 模型 / model\n' "${kind^^}"
        for ((i=0;i<${#nearby[@]};i++)); do printf ' [%d] %s\n' "$((i+1))" "${nearby[i]}"; done
        printf ' [D] 下载模型或组件 / Download models or components\n [R] 显示官方下载链接 / Show official download links\n [N] 输入新路径 / Enter a new path\n'
        value=$saved; [[ -n $value || ${#nearby[@]} == 0 ]] || value=1
        [[ -n $value || ${#nearby[@]} != 0 ]] || value=D
        prompt '选择模型、命令或 GGUF 路径 / Select a model, command, or GGUF path' "$value" true; value=$REPLY
        if [[ ${value,,} == d || ${value,,} == download ]]; then
            case $kind in
                vision) download_bundle flashnext-vision || continue ;;
                embedding) download_bundle qwen3-embedding-06b-q8 || continue ;;
                *) download_menu || continue ;;
            esac
            CANDIDATES=("${nearby[@]}"); catalog_candidates "$kind"; nearby=("${CANDIDATES[@]}")
            saved=''; continue
        fi
        if [[ ${value,,} == r || ${value,,} == recommended ]]; then recommendations "$kind"; continue; fi
        if [[ ${value,,} == n || ${value,,} == new ]]; then prompt 'GGUF 文件或目录 / GGUF file or directory' '' true; value=$REPLY; fi
        if valid_uint "$value" 1 "${#nearby[@]}"; then value=${nearby[10#$value-1]}; fi
        if [[ $value == \"*\" ]]; then value=${value:1:${#value}-2}; fi
        absolute "$value"; path=$REPLY
        if [[ -f $path && ${path,,} == *.gguf ]]; then S[$key]=$path; return; fi
        if [[ -d $path ]]; then
            model_candidates "$path" "$kind"
            if ((${#CANDIDATES[@]}==1)); then S[$key]=${CANDIDATES[0]}; return; fi
            printf '目录中有 %d 个候选文件，请输入准确文件路径。Directory has %d candidates; enter the exact file path.\n' "${#CANDIDATES[@]}" "${#CANDIDATES[@]}"; continue
        fi
        if [[ $kind == llm ]] && hub_reference "$value"; then S[$key]=$value; return; fi
        printf '未找到 GGUF 文件。GGUF file not found.\n'
    done
}

remember_model() {
    local current=${S[model]} value path count=0 history='' found
    [[ -n $current ]] || return 0
    if hub_reference "$current"; then return 0; fi
    absolute "$current"; current=$REPLY
    [[ -f $current && ${current,,} == *.gguf ]] || return 0
    while IFS= read -r value; do
        [[ -n $value ]] || continue; absolute "$value"; path=$REPLY
        [[ -f $path && ${path,,} == *.gguf ]] || continue
        found=false
        while IFS= read -r existing; do [[ $existing != "$path" ]] || found=true; done <<< "$history"
        [[ $found == false ]] || continue
        if [[ -n $history ]]; then history+=$'\n'; fi; history+="$path"; ((count+=1))
        ((count<20)) || break
    done <<< "$current"$'\n'"${S[model_history]}"
    S[model_history]=$history
}

# Stream bounded GGUF metadata; do not scan tensor weights or create a GPU context.
reasoning_efforts() (
    set +o pipefail
    od -An -v -tu1 -N 67108864 -- "$1" | LC_ALL=C awk '
    function refill( n) { if((getline line)<=0)exit 1;n=split(line,b);at=1;return n }
    function byte() { if(at>count)count=refill();used++;if(used>67108864)exit 1;return b[at++] }
    function uint(n, v,m,i) { v=0;m=1;for(i=0;i<n;i++){v+=byte()*m;m*=256}return v }
    function skip(n, take) { if(n<0||n>67108864-used)exit 1;while(n>0){if(at>count)count=refill();take=count-at+1;if(take>n)take=n;at+=take;n-=take;used+=take} }
    function str( n,i,v) { n=uint(8);if(n>16777216)exit 1;v="";for(i=0;i<n;i++)v=v sprintf("%c",byte());return v }
    function value(t,depth, e,n,i,size) {
        if(depth>8)exit 1
        if(t==8){skip(uint(8));return}
        if(t==9){e=uint(4);n=uint(8);if(n>1000000)exit 1;for(i=0;i<n;i++)value(e,depth+1);return}
        size=(t==0||t==1||t==7)?1:(t==2||t==3)?2:(t==4||t==5||t==6)?4:(t==10||t==11||t==12)?8:0
        if(!size)exit 1;skip(size)
    }
    BEGIN {
        at=1;count=0
        if(uint(4)!=1179993927)exit 1;v=uint(4);if(v!=2&&v!=3)exit 1
        uint(8);n=uint(8);if(n>1000000)exit 1
        for(i=0;i<n;i++){
            key=str();t=uint(4)
            if(t==8&&key=="general.architecture"){arch=str();have_arch=1}
            else if(t==8&&key=="tokenizer.chat_template"){template=str();have_template=1}
            else value(t,0)
            if(have_arch&&have_template)break
        }
        if((arch=="qwen35"||arch=="qwen35moe")&&template~/(^|[^a-zA-Z0-9_])enable_thinking([^a-zA-Z0-9_]|$)/)print "low medium high max"
        else if(template~/(^|[^a-zA-Z0-9_])reasoning_effort([^a-zA-Z0-9_]|$)/){if(arch=="qwen4exp")print "low medium xhigh";else print "low medium high max"}
        exit 0
    }'
)
physical_cores() {
    local path package core line value range first last cpu i
    local -A cores=() allowed=(); local -a ranges
    if [[ -r /proc/self/status ]]; then
        while IFS= read -r line; do
            [[ $line == Cpus_allowed_list:* ]] || continue
            trim "${line#*:}"; value=$REPLY; IFS=, read -r -a ranges <<< "$value"
            for range in "${ranges[@]}"; do
                first=${range%-*}; last=${range#*-}
                valid_uint "$first" 0 262143 && valid_uint "$last" "$first" 262143 || continue
                for ((i=first;i<=last;i++)); do allowed[$i]=1; done
            done
            break
        done < /proc/self/status
    fi
    for path in /sys/devices/system/cpu/cpu[0-9]*/topology; do
        [[ -r $path/core_id && -r $path/physical_package_id ]] || continue
        cpu=${path%/topology}; cpu=${cpu##*/cpu}
        [[ ${#allowed[@]} == 0 || -v 'allowed[$cpu]' ]] || continue
        if [[ -f ${path%/topology}/online && $(< "${path%/topology}/online") == 0 ]]; then continue; fi
        package=$(< "$path/physical_package_id"); core=$(< "$path/core_id"); cores["$package:$core"]=1
    done
    REPLY=${#cores[@]}; ((REPLY>0)) || REPLY=$(getconf _NPROCESSORS_ONLN); ((REPLY>0)) || REPLY=1
}
choose_device() {
    local output line device name kind mark default='' entry found=false; local -a options=()
    output=$("$BINARY" devices) || {
        fail $'Vulkan 设备枚举失败。请检查上方信息，再运行 `./infr devices`。\nVulkan device enumeration failed. Check the messages above, then run `./infr devices`.\nIntel/AMD Linux GPU 请安装或更新匹配的 Mesa Vulkan 驱动；混用新旧 Mesa 库也会导致此问题。\nOn Intel/AMD Linux GPUs, install or update the matching Mesa Vulkan driver; mixed old/new Mesa libraries can also cause this failure.'
        return 1
    }
    while IFS= read -r line; do
        if [[ $line =~ ^[[:space:]]*(Vulkan[0-9]+):[[:space:]]+(.+)[[:space:]]+\[([^]]+)\](.*)$ ]]; then
            device=${BASH_REMATCH[1]}; name=${BASH_REMATCH[2]}; kind=${BASH_REMATCH[3]}; mark=${BASH_REMATCH[4]}
            [[ ${kind,,} != *cpu* ]] || continue
            options+=("$device|$device: $name [$kind]"); [[ $mark != *default* ]] || default=$device
        fi
    done <<< "$output"
    ((${#options[@]}>0)) || {
        fail $'未找到可用的 Vulkan GPU。请运行 `./infr devices` 和 `vulkaninfo --summary` 检查系统驱动。\nNo usable Vulkan GPU was found. Run `./infr devices` and `vulkaninfo --summary` to inspect the system driver.\n“Driver does not support the ... PCI ID”警告表示 Mesa/Intel 驱动不匹配，并非模型路径错误。\nA "Driver does not support the ... PCI ID" warning is a Mesa/Intel driver mismatch, not a model-path error.'
        return 1
    }
    [[ -n $default ]] || default=${options[0]%%|*}
    for entry in "${options[@]}"; do [[ ${S[device]} != "${entry%%|*}" ]] || found=true; done
    [[ $found == true ]] || S[device]=$default
    choice device '设备 / Runtime device' "${options[@]}"
}
valid_address() {
    local host port part compressed=false; local -a parts
    if [[ $1 =~ ^\[([0-9a-fA-F:]+)\]:([0-9]+)$ ]]; then
        host=${BASH_REMATCH[1]}; port=${BASH_REMATCH[2]}; [[ $host == *:* && $host != *:::* ]] || return 1
        if [[ $host == *::* ]]; then
            part=${host#*::}; [[ $part != *::* ]] || return 1
            compressed=true; host=${host/::/:}; host=${host#:}; host=${host%:}
        else [[ $host != :* && $host != *: ]] || return 1; fi
        IFS=: read -r -a parts <<< "$host"
        for part in "${parts[@]}"; do [[ $part =~ ^[0-9a-fA-F]{1,4}$ ]] || return 1; done
        if [[ $compressed == true ]]; then ((${#parts[@]}<8)) || return 1; else ((${#parts[@]}==8)) || return 1; fi
    elif [[ $1 =~ ^([^:]+):([0-9]+)$ ]]; then
        host=${BASH_REMATCH[1]}; port=${BASH_REMATCH[2]}
        if [[ $host != localhost ]]; then
            [[ $host =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || return 1
            IFS=. read -r -a parts <<< "$host"; for part in "${parts[@]}"; do valid_uint "$part" 0 255 || return 1; done
        fi
    else return 1; fi
    valid_uint "$port" 1 65535
}
loopback() { [[ $1 =~ ^(127\.[0-9]+\.[0-9]+\.[0-9]+|localhost|\[::1\]):[0-9]+$ ]]; }
serial_mtp() { [[ ${S[mtp_enabled]} == true && ${S[server_parallel]} == 1 && ${S[server_vision]} == false && ${S[server_embedding]} == false ]]; }

newest_error_report() {
    local root=$ROOT/diagnostics/reports
    [[ -d $root ]] || { REPLY=''; return; }
    REPLY=$(find "$root" -type f -name '*.m4e' -print 2>/dev/null | LC_ALL=C sort | tail -n 1)
}

newest_benchmark_report() {
    local root=$ROOT/benchmark-database
    [[ -d $root ]] || { REPLY=''; return; }
    REPLY=$(find "$root" -type f -name '*.m4b' -print 2>/dev/null | LC_ALL=C sort | tail -n 1)
}

file_sha256() {
    REPLY=$(sha256sum -- "$1") || return 1
    REPLY=${REPLY%% *}
}

submission_header() {
    local path=$1 name=$2 count
    count=$(grep -c "^${name}: " "$path" 2>/dev/null) || true
    [[ $count == 1 ]] || return 1
    REPLY=$(sed -n "s/^${name}: //p" "$path")
}

ensure_submission_curl() {
    command -v curl >/dev/null 2>&1 && return 0
    if [[ ${INTERACTIVE-false} != true || ${HEADLESS-false} == true ]]; then
        fail 'Uploading requires curl. The local report was preserved.'
        return 1
    fi
    yes '上传需要 curl。现在使用系统包管理器安装？/ Uploading requires curl. Install it now?' false
    if [[ $REPLY != true ]]; then
        printf 'Upload skipped; the local report was preserved.\n'
        return 1
    fi
    install_curl
}

submission_api_config() {
    local origin_rest
    SUBMISSION_API_BASE=${MOE4ALL_SUBMISSION_API_BASE:-https://api.moe4all.com/v1}
    case $SUBMISSION_API_BASE in
        https://[A-Za-z0-9.-]*/v1|https://[A-Za-z0-9.-]*:[0-9]*/v1) ;;
        http://127.0.0.1:[0-9]*/v1|http://localhost:[0-9]*/v1)
            [[ -n ${MOE4ALL_SUBMISSION_API_BASE-} ]] || return 1
            ;;
        *) fail 'Invalid submission API base. HTTPS is required except for an explicit loopback development URL.'; return 1 ;;
    esac
    if [[ $SUBMISSION_API_BASE == *'?'* || $SUBMISSION_API_BASE == *'#'* || $SUBMISSION_API_BASE == *'@'* ]]; then
        fail 'Submission API base must not contain credentials, query or fragment.'
        return 1
    fi
    if [[ -n ${MOE4ALL_SUBMISSION_SITE_ORIGIN-} ]]; then
        SUBMISSION_SITE_ORIGIN=$MOE4ALL_SUBMISSION_SITE_ORIGIN
    elif [[ $SUBMISSION_API_BASE == https://api.moe4all.com/v1 ]]; then
        SUBMISSION_SITE_ORIGIN=https://moe4all.com
    else
        SUBMISSION_SITE_ORIGIN=${SUBMISSION_API_BASE%/v1}
    fi
    case $SUBMISSION_SITE_ORIGIN in
        https://[A-Za-z0-9.-]*|https://[A-Za-z0-9.-]*:[0-9]*|http://127.0.0.1:[0-9]*|http://localhost:[0-9]*) ;;
        *) fail 'Invalid submission website origin.'; return 1 ;;
    esac
    origin_rest=${SUBMISSION_SITE_ORIGIN#*://}
    if [[ $origin_rest == */* || $origin_rest == *'@'* || $SUBMISSION_SITE_ORIGIN == *'?'* || $SUBMISSION_SITE_ORIGIN == *'#'* ]]; then
        fail 'Submission website origin must not contain a path, credentials, query or fragment.'
        return 1
    fi
}

submission_api_request() {
    local method=$1 url=$2 output=$3 token=${4-} body=${5-} content_type=${6-application/json}
    local attempt code delay retry_after headers retry_epoch now_epoch
    mkdir -p -- "$CONFIG_DIR"
    headers=$(mktemp "$CONFIG_DIR/.submission-headers.XXXXXX") || return 1
    local -a args=(--silent --show-error --connect-timeout 10 --max-time 30 --request "$method" --header 'Accept: application/json' --dump-header "$headers")
    [[ -z $token ]] || args+=(--header "Authorization: Bearer $token")
    [[ -z $body ]] || args+=(--header "Content-Type: $content_type" --data-binary "@$body")
    for attempt in 0 1 2 3; do
        : > "$headers"
        code=$(submission_curl "${args[@]}" --output "$output" --write-out '%{http_code}' "$url") || code=000
        if [[ $code =~ ^2[0-9][0-9]$ ]]; then rm -f -- "$headers"; SUBMISSION_HTTP_STATUS=$code; return 0; fi
        if [[ $code != 000 && $code != 429 && ! $code =~ ^5[0-9][0-9]$ ]]; then break; fi
        ((attempt < 3)) || break
        delay=$((1 << (attempt + 1)))
        retry_after=$(awk 'tolower($0) ~ /^retry-after:/ { sub(/^[^:]*:[[:space:]]*/, ""); sub(/\r$/, ""); value=$0 } END { print value }' "$headers")
        if [[ $retry_after =~ ^[0-9]+$ ]]; then
            if (("${#retry_after}" > 3)); then break; fi
            delay=$((10#$retry_after))
        elif [[ -n $retry_after ]]; then
            retry_epoch=$(date -u -d "$retry_after" +%s 2>/dev/null) || retry_epoch=$(date -j -u -f '%a, %d %b %Y %T %Z' "$retry_after" +%s 2>/dev/null) || retry_epoch=''
            if [[ $retry_epoch =~ ^[0-9]+$ ]]; then
                now_epoch=$(date -u +%s); delay=$((retry_epoch - now_epoch)); ((delay >= 0)) || delay=0
            fi
        fi
        ((delay <= 120)) || break
        sleep "$delay"
    done
    rm -f -- "$headers"
    SUBMISSION_HTTP_STATUS=$code
    fail "Submission service request failed (HTTP $code). The local report was preserved."
    return 1
}

open_submission_url() {
    if command -v xdg-open >/dev/null 2>&1; then xdg-open "$1" >/dev/null 2>&1 & return 0; fi
    if command -v gio >/dev/null 2>&1; then gio open "$1" >/dev/null 2>&1 & return 0; fi
    return 1
}

save_submission_receipt() {
    local receipt_id=$1 receipt_token=$2 kind=$3 report_id=$4 file_hash=$5 payload_hash=$6 status=$7 received_at=$8 public_url=${9-}
    local directory=$CONFIG_DIR/private/submission-receipts path temporary
    path=$directory/$receipt_id.private-receipt.json
    mkdir -p -- "$directory"
    chmod 700 "$CONFIG_DIR/private" "$directory" 2>/dev/null || true
    temporary=$(mktemp "$directory/.receipt.XXXXXX")
    {
        printf '{\n'
        json_quote "$SUBMISSION_API_BASE"; printf '  "api_version": 1,\n  "api_base": %s,\n' "$REPLY"
        json_quote "$receipt_id"; printf '  "receipt_id": %s,\n' "$REPLY"
        json_quote "$receipt_token"; printf '  "receipt_token": %s,\n' "$REPLY"
        json_quote "$kind"; printf '  "kind": %s,\n' "$REPLY"
        json_quote "$report_id"; printf '  "report_id": %s,\n' "$REPLY"
        json_quote "$file_hash"; printf '  "file_sha256": %s,\n' "$REPLY"
    json_quote "${10-$file_hash}"; printf '  "submitted_file_sha256": %s,\n' "$REPLY"
        json_quote "$payload_hash"; printf '  "payload_sha256": %s,\n' "$REPLY"
        json_quote "$status"; printf '  "status": %s,\n' "$REPLY"
        json_quote "$received_at"; printf '  "received_at": %s,\n' "$REPLY"
        json_quote "$public_url"; printf '  "public_url": %s\n}\n' "$REPLY"
    } > "$temporary"
    chmod 600 "$temporary"
    mv -f -- "$temporary" "$path"
    REPLY=$path
}

submission_summary_json() {
    local path=$1 line value count=0
    printf '['
    while IFS= read -r line || [[ -n $line ]]; do
        case $line in
            '-----BEGIN MOE4ALL PAYLOAD-----') break ;;
            summary-[0-9][0-9][0-9]:\ *)
                value=${line#*: }; value=${value%$'\r'}
                (("${#value}" <= 500)) || continue
                ((count == 0)) || printf ','
                json_quote "$value"; printf '%s' "$REPLY"
                count=$((count + 1)); ((count < 8)) || break
                ;;
        esac
    done < "$path"
    printf ']'
}



pending_submission_dir() { REPLY="$CONFIG_DIR/private/pending-submissions"; }

submission_gh() (
    gh "$@"
)

submission_curl() {
    curl "$@"
}

queue_pending_submission() {
    local path=$1 kind=$2 directory destination temporary
    "$BINARY" submission verify "$path" >/dev/null 2>&1 || return 1
    pending_submission_dir; directory=$REPLY
    mkdir -p -- "$directory"; chmod 700 "$CONFIG_DIR/private" "$directory" 2>/dev/null || true
    destination="$directory/${path##*/}"
    if [[ $path != "$destination" ]]; then
        if [[ -f $destination ]] && ! cmp -s "$path" "$destination"; then
            fail '待发目录已有同 ID 的不同报告，已保留原件。A different report with this ID is already queued.'
            return 1
        fi
        temporary="$destination.moe4all-new-$$-$RANDOM"
        cp -- "$path" "$temporary" || return 1
        chmod 600 "$temporary" 2>/dev/null || true
        mv -f -- "$temporary" "$destination"
    fi
    [[ ${3-false} == true ]] || printf '提交暂存，待下次启动重试 / Submission saved for a later retry: %s\n' "$destination" >&2
}

retry_pending_submissions() {
    [[ ${PENDING_SUBMISSION_PROMPTED-false} != true ]] || return 0
    pending_submission_dir; local directory=$REPLY path kind retry
    local -a pending=()
    [[ -d $directory ]] || { PENDING_SUBMISSION_PROMPTED=true; return 0; }
    while IFS= read -r path; do pending+=("$path"); done < <(find "$directory" -type f \( -name '*.m4b' -o -name '*.m4e' \) -print 2>/dev/null | sort)
    ((${#pending[@]})) || { PENDING_SUBMISSION_PROMPTED=true; return 0; }
    PENDING_SUBMISSION_PROMPTED=true
    yes "发现 ${#pending[@]} 份待发报告，当前重试？/ Found ${#pending[@]} pending report(s). Retry now?" "${S[submission_pending_retry]-true}" || return 1
    retry=$REPLY; S[submission_pending_retry]=$retry; save_state
    [[ $retry == true ]] || return 0
    for path in "${pending[@]}"; do
        kind=benchmark; [[ $path != *.m4e ]] || kind=error
        if offer_submission_upload "$path" "$kind" true; then rm -f -- "$path"; fi
    done
}


website_submission_upload() {
    local path=$1 kind=$2 extension=.m4b size file_hash report_id response request
    local session_id session_token verification_url state='' receipt_id receipt_token payload_hash status received_at public_url response_hash response_size count original_hash summary_json
    [[ $kind != error ]] || extension=.m4e
    [[ -f $path && $path == *$extension ]] || { fail "Submission file is missing or does not match $kind."; return 1; }
    "$BINARY" submission verify "$path" >/dev/null || { fail 'Local submission verification failed; nothing was uploaded.'; return 1; }
    size=$(wc -c < "$path")
    trim "$size"; size=$REPLY
    ((size > 0 && size <= 131072)) || { fail 'Submission must be between 1 and 131072 bytes.'; return 1; }
    submission_header "$path" kind; [[ $REPLY == "$kind" ]] || { fail 'Submission kind does not match the selected upload type.'; return 1; }
    submission_header "$path" id; report_id=$REPLY
    [[ ${path##*/} == "${report_id}${extension}" ]] || { fail 'Submission filename does not match its report ID.'; return 1; }
    file_sha256 "$path"; file_hash=$REPLY
    ensure_submission_curl || return 1
    submission_api_config || return 1
    mkdir -p -- "$CONFIG_DIR"
    request=$(mktemp "$CONFIG_DIR/.submission-request.XXXXXX")
    response=$(mktemp "$CONFIG_DIR/.submission-response.XXXXXX")
    json_quote "$kind"; local q_kind=$REPLY
    json_quote "$report_id"; local q_id=$REPLY
    json_quote "$file_hash"; local q_hash=$REPLY
    summary_json=$(submission_summary_json "$path")
    printf '{"kind":%s,"report_id":%s,"file_sha256":%s,"size_bytes":%s,"summary":%s}\n' "$q_kind" "$q_id" "$q_hash" "$size" "$summary_json" > "$request"
    printf '\n正在创建安全上传会话 / Creating a secure upload session...\n'
    if ! submission_api_request POST "$SUBMISSION_API_BASE/upload-sessions" "$response" '' "$request"; then rm -f -- "$request" "$response"; return 1; fi
    if ! json_load "$response"; then rm -f -- "$request" "$response"; fail 'Submission service returned invalid JSON.'; return 1; fi
    json_get '["api_version"]'; [[ $REPLY == 1 ]] || { rm -f -- "$request" "$response"; return 1; }
    json_get '["state"]'; [[ $REPLY == awaiting_verification ]] || { rm -f -- "$request" "$response"; return 1; }
    json_get '["session_id"]'; session_id=$REPLY; [[ $session_id =~ ^[A-Za-z0-9_-]+$ ]] || { rm -f -- "$request" "$response"; return 1; }
    json_get '["session_token"]'; session_token=$REPLY; [[ -n $session_token ]] || { rm -f -- "$request" "$response"; return 1; }
    json_get '["max_upload_bytes"]'; response_size=$REPLY; [[ $response_size =~ ^[0-9]+$ ]] && ((response_size >= size)) || { rm -f -- "$request" "$response"; return 1; }
    json_get '["verification_url"]'; verification_url=$REPLY
    if [[ $verification_url != "$SUBMISSION_SITE_ORIGIN/community/upload.html#"* ]]; then rm -f -- "$request" "$response"; fail 'Submission verification URL is outside the configured website origin.'; return 1; fi
    if ! open_submission_url "$verification_url"; then
        printf 'Open this one-time confirmation URL on this machine:\n%s\n' "$verification_url"
    fi
    printf 'Confirm the public submission in the browser. The backend will create a public GitHub PR.\n'
    printf '请在浏览器确认公开提交；上传后后端会创建公开 GitHub PR。\n'
    for count in {1..300}; do
        sleep 3
        if ! submission_api_request GET "$SUBMISSION_API_BASE/upload-sessions/$session_id" "$response" "$session_token"; then rm -f -- "$request" "$response"; return 1; fi
        json_load "$response" || { rm -f -- "$request" "$response"; return 1; }
        json_get '["kind"]'; [[ $REPLY == "$kind" ]] || { rm -f -- "$request" "$response"; return 1; }
        json_get '["report_id"]'; [[ $REPLY == "$report_id" ]] || { rm -f -- "$request" "$response"; return 1; }
        json_get '["file_sha256"]'; [[ $REPLY == "$file_hash" ]] || { rm -f -- "$request" "$response"; return 1; }
        json_get '["size_bytes"]'; [[ $REPLY == "$size" ]] || { rm -f -- "$request" "$response"; return 1; }
        json_get '["state"]'; state=$REPLY
        case $state in
            ready|uploaded) break ;;
            cancelled) rm -f -- "$request" "$response"; printf 'Upload cancelled; the local report was preserved.\n'; return 2 ;;
            awaiting_verification) ;;
            *) rm -f -- "$request" "$response"; fail "Unknown upload session state: $state"; return 1 ;;
        esac
    done
    if [[ $state != ready && $state != uploaded ]]; then rm -f -- "$request" "$response"; fail 'Upload confirmation timed out; the local report was preserved.'; return 1; fi
    if ! submission_api_request POST "$SUBMISSION_API_BASE/upload-sessions/$session_id/report" "$response" "$session_token" "$path" 'text/plain; charset=utf-8'; then rm -f -- "$request" "$response"; return 1; fi
    json_load "$response" || { rm -f -- "$request" "$response"; return 1; }
    json_get '["receipt_id"]'; receipt_id=$REPLY; [[ $receipt_id =~ ^[A-Za-z0-9_-]+$ ]] || { rm -f -- "$request" "$response"; return 1; }
    json_get '["receipt_token"]'; receipt_token=$REPLY; [[ -n $receipt_token ]] || { rm -f -- "$request" "$response"; return 1; }
    json_get '["kind"]'; [[ $REPLY == "$kind" ]] || { rm -f -- "$request" "$response"; return 1; }
    json_get '["report_id"]'; [[ $REPLY == "$report_id" ]] || { rm -f -- "$request" "$response"; return 1; }
    json_get '["file_sha256"]'; original_hash=$REPLY; response_hash=$REPLY;
    local submitted_path='["submitted_file_sha256"]'
    if [[ -v 'J[$submitted_path]' ]]; then json_get '["submitted_file_sha256"]'; response_hash=$REPLY; fi
    [[ $response_hash == "$file_hash" ]] || { rm -f -- "$request" "$response"; return 1; }
    json_get '["payload_sha256"]'; payload_hash=$REPLY; [[ $payload_hash =~ ^[0-9a-f]{64}$ ]] || { rm -f -- "$request" "$response"; return 1; }
    json_get '["status"]'; status=$REPLY
    case $status in forwarding|pending_review|published|rejected|delivery_failed) ;; *) rm -f -- "$request" "$response"; fail "Unknown receipt status: $status"; return 1 ;; esac
    json_get '["received_at"]'; received_at=$REPLY
    public_url=''
    local public_path='["public_url"]'
    if [[ -v 'J[$public_path]' && ${J[$public_path]} == \"*\" ]]; then json_get "$public_path"; public_url=$REPLY; fi
    save_submission_receipt "$receipt_id" "$receipt_token" "$kind" "$report_id" "$original_hash" "$payload_hash" "$status" "$received_at" "$public_url" "$file_hash"
    printf '已收到，状态 / Received, status: %s\n回执 ID / Receipt ID: %s\n私有回执 / Private receipt: %s\n' "$status" "$receipt_id" "$REPLY"
    [[ -z $public_url ]] || printf 'Public URL: %s\n' "$public_url"
    [[ -n $public_url ]] || printf '需要查看进度时，在网站“我的回执”导入保存的私密回执文件。\nTo check progress, import the saved private receipt under My receipts on the website.\n'
    case $status in
        delivery_failed) printf '转发已失败，云端正文已清理；请重新导出新报告。Delivery failed; export a new report to try again.\n' >&2 ;;
        rejected) printf '该报告 PR 已退回。The report PR was closed without merging.\n' >&2 ;;
    esac
    rm -f -- "$request" "$response"
}

gh_submission_upload() {
    local path=$1 kind=$2 retry=${3-1} login upstream=Headmaster218/MoE4All fork month report_id branch base_sha relative request content pr_url file_hash existing
    command -v gh >/dev/null 2>&1 || return 1
    submission_gh auth status --hostname github.com >/dev/null 2>&1 || return 1
    "$BINARY" submission verify "$path" >/dev/null || return 3
    login=$(submission_gh api user --jq .login 2>/dev/null) || return 1
    [[ $login =~ ^[A-Za-z0-9-]+$ ]] || return 1
    submission_header "$path" kind; [[ $REPLY == "$kind" ]] || return 1
    submission_header "$path" id; report_id=$REPLY
    submission_header "$path" created-at; month=${REPLY:0:7}
    [[ $month =~ ^[0-9]{4}-[0-9]{2}$ ]] || return 1
    fork=$login/MoE4All
    relative=$kind-database/v1/$month/${path##*/}
    printf '\nGitHub: %s\n目标 / Target: %s/%s\n' "$login" "$upstream" "$relative"
    printf 'PR 创建后报告即公开；不创建 Issue，不自动合并。\nThe report becomes public when the PR is created; no Issue or automatic merge.\n'
    file_sha256 "$path"; file_hash=$REPLY
    if [[ $retry != 2 || ${SUBMISSION_GH_CONFIRMED-} != "$kind/$file_hash" ]]; then
        yes '现在创建分支和报告 PR？/ Create the branch and report PR now?' true || return 2
        [[ $REPLY == true ]] || return 2
        SUBMISSION_GH_CONFIRMED="$kind/$file_hash"
    fi
    if ! submission_gh api "repos/$fork" >/dev/null 2>&1; then
        submission_gh repo fork "$upstream" --clone=false --remote=false >/dev/null || return 1
        local ready=false attempt
        for attempt in {1..15}; do sleep 2; if submission_gh api "repos/$fork" >/dev/null 2>&1; then ready=true; break; fi; done
        [[ $ready == true ]] || return 1
    fi
    base_sha=$(submission_gh api "repos/$upstream/git/ref/heads/main" --jq .object.sha 2>/dev/null) || return 1
    [[ $base_sha =~ ^[a-f0-9]{40}$ ]] || return 1
    file_sha256 "$path"; file_hash=$REPLY
    branch=community/$kind/$file_hash
    if ! submission_gh api --method POST "repos/$fork/git/refs" -f "ref=refs/heads/$branch" -f "sha=$base_sha" >/dev/null 2>&1; then
        submission_gh api "repos/$fork/git/ref/heads/$branch" >/dev/null 2>&1 || return 1
    fi
    mkdir -p -- "$CONFIG_DIR"
    request=$(mktemp "$CONFIG_DIR/.github-submission.XXXXXX")
    content=$(base64 < "$path" | tr -d '\r\n') || { rm -f -- "$request"; return 1; }
    json_quote "$kind: add $report_id"; local q_message=$REPLY
    json_quote "$content"; local q_content=$REPLY
    json_quote "$branch"; local q_branch=$REPLY
    printf '{"message":%s,"content":%s,"branch":%s}\n' "$q_message" "$q_content" "$q_branch" > "$request"
    if ! submission_gh api --method PUT "repos/$fork/contents/$relative" --input "$request" >/dev/null 2>&1; then
        existing=$(submission_gh api "repos/$fork/contents/$relative?ref=$branch" --jq .content 2>/dev/null | tr -d '\r\n') || { rm -f -- "$request"; return 1; }
        if [[ $existing != "$content" ]]; then rm -f -- "$request"; fail 'Existing report differs; nothing was overwritten.'; return 1; fi
    fi
    rm -f -- "$request"
    pr_url=$(submission_gh pr create --repo "$upstream" --base main --head "$login:$branch" --title "$kind: add $report_id" --body 'User-consented public MoE4All report. Awaiting maintainer review; no automatic merge. Format validation is not an independent reproduction.' 2>/dev/null) ||
        pr_url=$(submission_gh pr list --repo "$upstream" --head "$login:$branch" --state all --json url --jq '.[0].url // empty' 2>/dev/null) || return 1
    [[ $pr_url =~ ^https://github.com/Headmaster218/MoE4All/pull/[0-9]+$ ]] || return 1
    printf 'PR 已创建或已存在 / PR created or already exists: %s\n' "$pr_url"
}

gh_benchmark_upload() { gh_submission_upload "$1" benchmark; }

submission_urlencode() {
    local hex pair encoded=''
    hex=$(printf '%s' "$1" | od -An -v -tx1 | tr -d ' \n') || return 1
    while [[ -n $hex ]]; do pair=${hex:0:2}; encoded="$encoded%$pair"; hex=${hex:2}; done
    REPLY=$encoded
}

copy_submission_text() {
    if command -v pbcopy >/dev/null 2>&1 && pbcopy < "$1" 2>/dev/null; then return 0; fi
    if command -v wl-copy >/dev/null 2>&1 && wl-copy < "$1" 2>/dev/null; then return 0; fi
    if command -v xclip >/dev/null 2>&1 && xclip -selection clipboard < "$1" 2>/dev/null; then return 0; fi
    if command -v xsel >/dev/null 2>&1 && xsel --clipboard --input < "$1" 2>/dev/null; then return 0; fi
    return 1
}

open_github_submission() {
    local path=$1 kind=$2 month report_id file_hash guidance copied=false filename message content url
    "$BINARY" submission verify "$path" >/dev/null || return 1
    submission_header "$path" kind; [[ $REPLY == "$kind" ]] || return 1
    submission_header "$path" id; report_id=$REPLY
    submission_header "$path" created-at; month=${REPLY:0:7}
    [[ $month =~ ^[0-9]{4}-[0-9]{2}$ && $report_id =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$ ]] || return 1
    file_sha256 "$path"; file_hash=$REPLY
    copy_submission_text "$path" && copied=true
    guidance=$(printf '%s\n' \
        '上传步骤：' \
        '1. 在这个编辑框内按 Ctrl+A，全选并删除本提示。' \
        '2. 按 Ctrl+V，粘贴完整的 .m4b 或 .m4e 报告文本。' \
        '3. 点击右上角 Commit changes...（无写权限时可能显示 Propose changes）。' \
        '4. 选择新建分支（不要直接提交 main），确认提交，然后点击 Create pull request。' \
        '5. 检查标题与文件后，再点击 Create pull request，完成上传。' \
        '' 'Upload steps:' \
        '1. Press Ctrl+A in this editor to select and remove this instruction.' \
        '2. Press Ctrl+V to paste the complete .m4b or .m4e report text.' \
        '3. Click Commit changes... at the upper right (or Propose changes).' \
        '4. Choose a new branch (not a direct commit to main), confirm, then click Create pull request.' \
        '5. Review the title and file, then click Create pull request again to finish.' \
        '' 'macOS 请使用 Cmd+A / Cmd+V。On macOS, use Cmd+A / Cmd+V.' "建议分支名 / Suggested branch: community/$kind/$file_hash")
    if [[ $copied != true ]]; then
        guidance=$(printf '%s\n\n%s' '剪贴板复制未成功：先打开本地报告，全选并复制完整文本。Clipboard copy failed: open the local report and copy its complete text first.' "$guidance")
    fi
    submission_urlencode "${path##*/}"; filename=$REPLY
    submission_urlencode "$kind: add $report_id"; message=$REPLY
    submission_urlencode "$guidance"; content=$REPLY
    url="https://github.com/Headmaster218/MoE4All/new/main/$kind-database/v1/$month?filename=$filename&message=$message&value=$content"
    printf '本地报告 / Local report: %s\n' "$path"
    if open_submission_url "$url"; then
        printf '已打开 GitHub 上传页，请按编辑区内的双语提示完成。Opened GitHub; follow the bilingual instructions in the editor.\n'
    else
        printf '未能自动打开 GitHub，请手动打开以下地址。Could not open GitHub; open this URL manually:\n%s\n' "$url"
        return 1
    fi
}

submission_next_action() {
    if [[ ${1-} == browser ]]; then
        case ${S[submission_browser_action]-later} in sent|reopen|cloudflare|later) ;; *) S[submission_browser_action]=later ;; esac
        choice submission_browser_action 'GitHub 网页提交 / GitHub web submission' \
            'sent|已成功提交 / Successfully submitted' \
            'reopen|重新打开 GitHub / Reopen GitHub' \
            'cloudflare|改用 Cloudflare / Use Cloudflare instead' \
            'later|下次再发 / Send later' || return 1
        save_state
        REPLY=${S[submission_browser_action]}
    else
        case ${S[submission_failure_action]-later} in retry|channel|later) ;; *) S[submission_failure_action]=later ;; esac
        choice submission_failure_action 'Cloudflare 上传未完成 / Cloudflare upload incomplete' \
            'retry|重试 Cloudflare / Retry Cloudflare' \
            'channel|返回渠道选择 / Return to channel selection' \
            'later|下次再发 / Send later' || return 1
        save_state
        REPLY=${S[submission_failure_action]}
    fi
}

offer_submission_upload() {
    local path=$1 kind=$2 consented=${3-false} has_github channel=choose attempt action consent_key=benchmark_upload consent
    "$BINARY" submission verify "$path" >/dev/null || return 1
    if [[ $consented != true ]]; then
        [[ $kind != error ]] || consent_key=error_upload
        yes '愿意公开提交这份报告吗？/ Submit this report publicly?' "${S[$consent_key]}" || return 1
        consent=$REPLY; S[$consent_key]=$consent; save_state
        [[ $consent == true ]] || return 1
    fi
    queue_pending_submission "$path" "$kind" true || return 1
    local pending_path="$CONFIG_DIR/private/pending-submissions/${path##*/}" upload_status=1
    while true; do
        if [[ $channel == choose ]]; then
            yes '你有 GitHub 账号吗？/ Do you have a GitHub account?' "${S[benchmark_upload_has_github]}" || return 1
            has_github=$REPLY; S[benchmark_upload_has_github]=$has_github; save_state
            channel=cloudflare; [[ $has_github != true ]] || channel=github
        fi
        if [[ $channel == github ]]; then
            SUBMISSION_GH_CONFIRMED=''
            if command -v gh >/dev/null 2>&1 && submission_gh auth status --hostname github.com >/dev/null 2>&1; then
                for attempt in 1 2; do
                    printf 'GitHub CLI 上传 / GitHub CLI upload: %s/2\n' "$attempt"
                    if gh_submission_upload "$path" "$kind" "$attempt"; then rm -f -- "$pending_path"; return 0; else upload_status=$?; fi
                    [[ $upload_status != 2 && $upload_status != 3 ]] || return 1
                done
            fi
            printf '没有可用的已登录 CLI 或上传未完成，改用 GitHub 网页。\nNo signed-in CLI is available or CLI upload did not complete; opening GitHub.\n'
            channel=browser
        fi
        if [[ $channel == browser ]]; then
            open_github_submission "$path" "$kind" || true
            printf '完成网页操作后再选择；向导不会查询网页提交结果。\nChoose after completing the web steps; the wizard does not query their result.\n'
            submission_next_action browser || return 1
            action=$REPLY
            case $action in
                sent) rm -f -- "$pending_path"; return 0 ;;
                reopen) continue ;;
                cloudflare) channel=cloudflare ;;
                *) break ;;
            esac
        fi
        if [[ $channel == cloudflare ]]; then
            if website_submission_upload "$path" "$kind"; then rm -f -- "$pending_path"; return 0; fi
            submission_next_action || return 1
            action=$REPLY
            case $action in retry) continue ;; channel) channel=choose; continue ;; *) break ;; esac
        fi
    done
    printf '提交暂存，待下次启动重试 / Submission saved for a later retry: %s\n' "$pending_path"
    return 1
}

offer_benchmark_upload() { offer_submission_upload "$1" benchmark; }

pack_manual_benchmark() {
    local raw_json=$1 payload output id created engine_version model_name model_size context_tokens avg result_path avg_path='[0,"avg_ts"]'
    json_load "$raw_json" || { fail 'Manual benchmark did not return recognizable JSON.'; return 1; }
    [[ $JSON_KIND == array && -v 'J[$avg_path]' ]] || { fail 'Manual benchmark JSON has no avg_ts result.'; return 1; }
    avg=${J[$avg_path]}
    [[ $avg =~ ^[0-9]+([.][0-9]+)?$ ]] || { fail 'Manual benchmark avg_ts is invalid.'; return 1; }
    created=$(date -u '+%Y-%m-%dT%H:%M:%S.000Z')
    id=$(date -u '+manual-%Y%m%dT%H%M%S.000Z-')$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
    engine_version=$("$BINARY" --version 2>/dev/null || true); engine_version=${engine_version##* }
    model_name=${S[model]##*/}; model_size=0; [[ ! -f ${S[model]} ]] || model_size=$(wc -c < "${S[model]}")
    context_tokens=${S[depth_tokens]}; [[ ${S[manual_bench_kind]} != prefill ]] || context_tokens=${S[prompt_tokens]}
    payload=$(mktemp "$CONFIG_DIR/.benchmark-payload.XXXXXX")
    json_quote "$id"; local q_id=$REPLY
    json_quote "$created"; local q_created=$REPLY
    json_quote "$engine_version"; local q_engine=$REPLY
    json_quote "$model_name"; local q_model=$REPLY
    json_quote "${S[device]}"; local q_device=$REPLY
    json_quote "${S[manual_bench_kind]}"; local q_kind=$REPLY
    json_quote "${S[depth_mode]}"; local q_depth=$REPLY
    json_quote "${S[setup_mode]}"; local q_profile=$REPLY
    cat > "$payload" <<EOF
{"schema_version":1,"id":$q_id,"created_at_utc":$q_created,"engine_version":$q_engine,"engine_commit":null,"profile":"manual","model":{"name":$q_model,"storage_bytes":$model_size},"hardware":{"os":"linux","architecture":"$(uname -m)","cpu":{},"ram":{},"gpu":{"configured_device":$q_device}},"configuration":{"benchmark_kind":$q_kind,"prompt_tokens":${S[prompt_tokens]},"decode_tokens":${S[gen_tokens]},"depth_mode":$q_depth,"depth_tokens":${S[depth_tokens]},"repetitions":${S[reps]},"auto_profile":$q_profile},"contexts":[{"requested_context_tokens":$context_tokens,"prefill":null,"decode":[],"manual_result":{"kind":$q_kind,"average_tokens_per_second":$avg},"failures":[]}],"failures":[]}
EOF
    if ! output=$("$BINARY" submission pack-benchmark --input "$payload" --result-dir "$ROOT/benchmark-database" 2>&1); then rm -f -- "$payload"; fail "$output"; return 1; fi
    rm -f -- "$payload"
    result_path=$(printf '%s\n' "$output" | sed -n 's/^MOE4ALL_SUBMISSION_PATH=//p' | tail -n 1)
    [[ -n $result_path ]] || { fail 'The engine did not return a benchmark submission path.'; return 1; }
    printf '\n结果 / Results\n+------------------+------------+----------+\n'
    printf '| Workload         | Repetitions| Overall  |\n+------------------+------------+----------+\n'
    printf '| %-16s | %10s | %8.1f |\n' "${S[manual_bench_kind]}" "${S[reps]}" "$avg"
    printf '+------------------+------------+----------+\nBenchmark submission: %s\n' "$result_path"
    REPLY=$result_path
}

read_problem_description() {
    local description='' mode=rewrite line added decision
    while true; do
        [[ $mode != rewrite ]] || description=''
        printf '\n请输入问题描述 / Describe the problem。\n可输入多行，单独输入 . 才结束 / Multiple lines are allowed; enter a single period . to finish.\n'
        added=''
        while true; do
            IFS= read -r line || return 1
            [[ $line != . ]] || break
            added+="$line"$'\n'
        done
        added=${added%$'\n'}
        if [[ $mode == append && -n $description && -n $added ]]; then description+=$'\n'$added
        elif [[ -n $added || $mode == rewrite ]]; then description=$added; fi
        printf '\n问题描述预览 / Description preview:\n%s\n' "${description:-(空 / empty)}"
        printf '1. 确认 / Confirm  2. 重写 / Rewrite  3. 追加 / Append  4. 取消 / Cancel [1]: '
        IFS= read -r decision || return 1; [[ -n $decision ]] || decision=1
        case $decision in 1) REPLY=$description; return ;; 2) mode=rewrite ;; 3) mode=append ;; 4) return 2 ;; *) printf '请选择 1-4。Choose 1-4.\n' ;; esac
    done
}

complete_error_report() {
    local source=${1-} description_file output path; local -a report_args
    read_problem_description || { printf '已取消；现有诊断文件未删除。Cancelled; existing diagnostics were not deleted.\n'; return 0; }
    mkdir -p -- "$CONFIG_DIR"
    description_file=$(mktemp "$CONFIG_DIR/.error-description.XXXXXX")
    printf '%s' "$REPLY" > "$description_file"
    report_args=(report)
    if [[ -n $source ]]; then report_args+=(--source "$source")
    else report_args+=(--diagnostics-dir "$ROOT/diagnostics"); fi
    report_args+=(--description-file "$description_file")
    if ! output=$("$BINARY" "${report_args[@]}" 2>&1); then rm -f -- "$description_file"; fail "$output"; return 1; fi
    rm -f -- "$description_file"
    printf '%s\n' "$output"
    path=$(printf '%s\n' "$output" | sed -n 's/^MOE4ALL_SUBMISSION_PATH=//p' | tail -n 1)
    [[ -n $path ]] || { fail 'The engine did not return an error-report path.'; return 1; }
    yes '愿意公开提交这份错误报告吗？/ Submit this error report publicly?' "${S[error_upload]}"; S[error_upload]=$REPLY
    save_state
    if [[ ${S[error_upload]} == true ]]; then
        offer_submission_upload "$path" error true || printf '上传未完成；本地报告仍保存在 %s。Upload did not complete; the local report remains there.\n' "$path" >&2
    fi
}

configure() {
    local key value effort count previous
    REPORT_RETURN_MODE=${S[launch_mode]}
    choice launch_mode '你想做什么？/ What would you like to do?' 'run|实时终端对话（推荐）/ Terminal chat (recommended)' 'serve|启动 OpenAI 兼容 API / Start OpenAI-compatible API server' 'bench|性能测试 / Benchmark' 'report|报告当前运行问题 / Report a current runtime problem'
    [[ ${S[launch_mode]} != report ]] || return 0
    model_path llm model
    if [[ ${S[launch_mode]} == serve ]]; then
        if [[ -z ${S[vision_projector]} ]]; then model_candidates "${S[model]%/*}" vision; ((${#CANDIDATES[@]}!=1)) || S[vision_projector]=${CANDIDATES[0]}; fi
        ask_yes server_vision '启用视觉图片理解？/ Enable image understanding?'; [[ ${S[server_vision]} != true ]] || model_path vision vision_projector
        ask_yes server_embedding '同时提供 Embedding API？/ Also serve the Embedding API?'
        if [[ ${S[server_embedding]} == true ]]; then model_path embedding embedding_model; integer embedding_idle_timeout 'Embedding 空闲释放秒数，0 为服务期间常驻 / Idle eviction seconds, 0 keeps resident'; fi
    fi
    choose_device
    choice setup_mode '配置方式 / Configuration' 'conservative|自动配置：保守（推荐）/ Automatic: conservative (recommended)' 'aggressive|自动配置：激进性能 / Automatic: aggressive performance' 'manual|全手动配置 / Fully manual configuration'
    if [[ ${S[setup_mode]} == manual ]]; then
        integer ubatch 'Ubatch，留空为自动 / Ubatch, blank for auto' 1 999999999 true; integer threads 'CPU 线程，留空为全部 / CPU threads, blank for all' 1 999999999 true
        while true; do ask config_path '配置 TOML，留空使用默认查找 / Config TOML, blank for default lookup'; [[ -n ${S[config_path]} ]] || break; absolute "${S[config_path]}"; [[ ! -f $REPLY ]] || break; printf '未找到配置 TOML。Config TOML not found.\n'; done
        choice kv_preset 'KV Cache 类型 / KV cache type' 'auto|引擎自动（推荐）/ Engine default (recommended)' 'q8|Q8_0 K + Q8_0 V' 'f16|F16 K + F16 V' 'custom|分别指定 / Custom K and V'
        case ${S[kv_preset]} in custom) ask kv_type_k 'K 类型 / K dtype' true; ask kv_type_v 'V 类型 / V dtype' true ;; q8) S[kv_type_k]=q8_0; S[kv_type_v]=q8_0 ;; f16) S[kv_type_k]=f16; S[kv_type_v]=f16 ;; esac
        ask_yes configure_memory '设置显存、内存和分页参数？/ Configure memory and paging?'
        if [[ ${S[configure_memory]} == true ]]; then
            ask vram_budget '总显存预算，留空为自动 / Total VRAM budget, blank for auto'
            ask vram_reserve '额外显存保留，留空为自动 / Additional VRAM reserve, blank for auto'
            ask expert_cache 'GPU 专家缓存，留空为自动 / GPU expert cache, blank for auto'
            ask ram_budget '进程总 RAM 预算，留空为自动 / Total process RAM budget, blank for auto'
            ask pager_ring 'Pager staging ring，留空为自动 / Pager staging ring, blank for auto'
            ask_yes host_dma '启用 RAM 到 VRAM Host DMA？/ Enable RAM-to-VRAM Host DMA?'; integer pager_ring_slots 'Pager ring slots，留空为默认 / Pager ring slots, blank for default' 2 999999999 true
            ask_yes kv_overflow '允许 KV 溢出到 RAM？/ Allow KV overflow to RAM?'
            if [[ ${S[kv_overflow]} == true ]]; then integer kv_overflow_vram_mb 'KV 溢出 VRAM MiB / KV overflow VRAM MiB' 1 999999999 true; integer kv_overflow_reserve_mb 'KV 溢出保留 MiB / KV overflow reserve MiB' 1 999999999 true; fi
        fi
        choice submit_mode 'Submit splitter' 'auto|自动反馈 / Automatic feedback' 'disabled|禁用，no-split / Disabled, no-split' 'fixed|固定 cap / Fixed cap'
        [[ ${S[submit_mode]} != fixed ]] || integer submit_cap '固定 dispatch cap / Fixed dispatch cap' 1
        ask_yes configure_diagnostics '设置统计或 profiler？/ Configure statistics or profilers?'
        if [[ ${S[configure_diagnostics]} == true ]]; then for key in pager_stats pager_profile stage_profile vram_profile; do ask_yes "$key" "$key"; done; fi
        ask custom_sets '额外 --set，以分号分隔 / Extra --set entries separated by semicolons'
    else S[kv_preset]=q8; S[kv_type_k]=q8_0; S[kv_type_v]=q8_0; fi
    S[auto_overrides]=''
    if [[ ${S[launch_mode]} != bench ]]; then
        ask_yes configure_thinking '是否配置默认思考？API 可覆盖 / Configure default reasoning? API may override'
        previous=${S[think_mode]}; effort=${S[reasoning_effort]}; S[think_mode]=default; S[reasoning_effort]=default
        if [[ ${S[configure_thinking]} == true ]]; then
            value=true; [[ $previous != no-think ]] || value=false; yes '默认开启思考？/ Enable reasoning by default?' "$value"
            if [[ $REPLY == true ]]; then
                S[think_mode]=think; value=$(reasoning_efforts "${S[model]}") || value=''
                if [[ -n $value ]]; then
                    local -a options=('default|模板默认 / Template default'); local found=false
                    for key in $value; do options+=("$key|$key"); [[ $key != "$effort" ]] || found=true; done
                    [[ $found == false ]] || S[reasoning_effort]=$effort
                    choice reasoning_effort '默认思考强度 / Default reasoning effort' "${options[@]}"
                else printf '未检测到思考强度支持，将保留模板默认值。Reasoning-effort support not detected; keeping template default.\n'; fi
            else S[think_mode]=no-think; fi
        fi
        integer max_new '每轮最大生成 token（思考与正文合计）/ Max generated tokens per reply (reasoning + answer)' 1
        ask_yes configure_sampling '设置默认采样参数？/ Configure default sampling?'
        if [[ ${S[configure_sampling]} == true ]]; then ask temperature '温度，留空为模型默认 / Temperature, blank for model default'; integer top_k 'Top K，留空为模型默认 / Top K, blank for model default' 0 999999999 true; ask top_p 'Top P，留空为模型默认 / Top P, blank for model default'; integer seed '随机种子，留空为模型默认 / Seed, blank for model default' 0 999999999 true; fi
    fi
    ask_yes cpu_miss_enabled '启用实验性 CPU 专家 miss 计算？/ Enable experimental CPU expert-miss computation?'
    if [[ ${S[cpu_miss_enabled]} == true ]]; then
        physical_cores; count=$REPLY
        printf '物理核心：%s。需要 AVX2 + FMA3 和完整 RAM 专家缓存。\nPhysical cores: %s. Requires AVX2 + FMA3 and full-RAM experts.\nLinux 使用操作系统调度，不使用 Windows 混合核心绑定。\nLinux uses OS scheduling, not Windows hybrid-core pinning.\n' "$count" "$count"
        integer cpu_miss_max '最多交给 CPU 的 miss 数（1-3）/ Maximum CPU miss count (1-3)' 1 3
        valid_uint "${S[cpu_miss_cores]}" 1 "$count" || S[cpu_miss_cores]=$((count>2 ? count-2 : 1))
        integer cpu_miss_cores 'CPU 计算核心数 / CPU compute cores' 1 "$count"
    fi
    if [[ ${S[launch_mode]} == serve ]]; then
        integer server_parallel '并发会话数（每个会话有独立 KV）/ Concurrent slots (one KV cache each)' 1
    fi
    if [[ ${S[launch_mode]} != bench ]]; then ask context '上下文窗口，留空为自动 / Context window, blank for auto'; fi
    if [[ ${S[launch_mode]} == serve ]]; then
        if serial_mtp; then S[server_session_cache]=false; printf '单路纯文本 MTP 会禁用 SSD 会话缓存。Single-stream text-only MTP disables SSD session caching.\n'
        else ask_yes server_session_cache '将闲置会话 KV 缓存到 SSD？/ Cache idle-session KV on SSD?'; fi
        if [[ ${S[server_session_cache]} == true ]]; then
            yes '修改 SSD KV 缓存参数？/ Customize SSD KV cache settings?' false
            if [[ $REPLY == true ]]; then ask session_cache_dir 'KV 缓存目录 / KV cache directory' true; integer session_idle_secs '会话闲置多少秒后写入 SSD / Spill after idle seconds'; ask session_cache_max 'SSD KV 缓存总上限（绝对大小）/ Total SSD KV cache limit (absolute size)' true; integer session_cache_ttl_hours '缓存保留小时数，0 为不按时间清理 / Cache TTL hours, 0 disables age expiry'; fi
        fi
        while true; do ask server_addr '监听地址（IP:端口）/ Listen address (IP:port)' true; valid_address "${S[server_addr]}" && break; printf '请输入 IP:端口，例如 127.0.0.1:8080 或 [::1]:8080。Enter an IP:port, such as 127.0.0.1:8080 or [::1]:8080.\n'; done
        ask_yes server_auth '启用 Bearer API key 鉴权？/ Enable Bearer API-key authentication?'
    fi
    if [[ ${S[launch_mode]} == bench ]]; then
        if [[ ! ${S[bench_kind]} =~ ^(quick|standard|full|manual)$ ]]; then S[manual_bench_kind]=${S[bench_kind]}; S[bench_kind]=manual; fi
        choice bench_kind '测试方案 / Benchmark plan' \
            $'quick|自动快速 / Automatic quick|6K Prefill + 1024 Decode。单发，一次。\n6K Prefill + 1024 Decode. x1, one run.' \
            $'standard|自动标准 / Automatic standard|6K Prefill + 1024 Decode；30K Prefill + 1024 Decode；1-2 并发，各一次，共 4 次。\n6K Prefill + 1024 Decode; 30K Prefill + 1024 Decode; x1-x2, one run each, 4 runs total.' \
            $'full|自动完整 / Automatic full|6K、30K、150K Prefill + 1024 Decode；1-4 并发，各三次，共 36 次。\n6K, 30K and 150K Prefill + 1024 Decode; x1-x4, three runs each, 36 runs total.' \
            $'manual|手动测试 / Manual benchmark|按顺序添加 Prefill、Decode 和上下文深度。\nBuild one test step by step: Prefill, Decode and context depth.'
        if [[ ${S[bench_kind]} != manual ]]; then
            case ${S[bench_kind]} in quick) S[bench_parallel]=1 ;; standard) S[bench_parallel]=2 ;; full) S[bench_parallel]=4 ;; esac
            integer bench_parallel '最大并发路数（会依次测试 x1 到 xN）/ Maximum concurrency (tests x1 through xN)' 1 4
        else
            [[ ${S[manual_bench_kind]} =~ ^(prefill|decode|mixed)$ ]] || S[manual_bench_kind]=decode
            choice manual_bench_kind '第一步：选择要添加的工作 / Step 1: choose work' \
                $'prefill|只测 Prefill / Prefill only|处理一段提示词，不生成 token。\nProcess a prompt without generating tokens.' \
                $'decode|只测 Decode / Decode only|生成 token；下一步可添加已有上下文。\nGenerate tokens; existing context can be added next.' \
                $'mixed|Prefill + Decode|同一轮先处理提示词，再生成 token。\nProcess a prompt and then generate in the same turn.'
            case ${S[manual_bench_kind]} in
                prefill) integer prompt_tokens 'Prefill token 数 / Prefill tokens' 1; S[gen_tokens]=0; S[depth_mode]=none; S[depth_tokens]=0 ;;
                decode) S[prompt_tokens]=0; integer gen_tokens 'Decode token 数 / Decode tokens' 1 ;;
                mixed) integer prompt_tokens 'Prefill token 数 / Prefill tokens' 1; printf '已添加 / Added: Prefill %s tokens\n' "${S[prompt_tokens]}"; integer gen_tokens 'Decode token 数 / Decode tokens' 1 ;;
            esac
            printf '已添加 / Added: Prefill %s, Decode %s tokens\n' "${S[prompt_tokens]}" "${S[gen_tokens]}"
            if [[ ${S[manual_bench_kind]} != prefill ]]; then
                choice depth_mode '添加测量前上下文 / Add context before measurement' \
                    $'none|不添加 / None|从空 KV 开始测量。\nMeasure from an empty KV cache.' \
                    $'real|真实上下文 / Real context|先真实 Prefill 指定长度，再开始计时。\nReally prefill the selected depth before timing.' \
                    $'synthetic|Synthetic KV 深度 / Synthetic KV depth|只构造 KV 深度与注意力形状，不计作真实 Prefill。\nCreate KV depth and attention shape without a real Prefill.'
                if [[ ${S[depth_mode]} == none ]]; then S[depth_tokens]=0; else integer depth_tokens '已有上下文 token 数 / Existing context tokens' 1; fi
                printf '已添加 / Added: context=%s, depth=%s\n' "${S[depth_mode]}" "${S[depth_tokens]}"
            fi
            integer reps '重复次数 / Repetitions' 1; S[json_output]=true
            printf '最终计划 / Final plan: Prefill %s, Decode %s, context %s/%s, repetitions %s。JSON 自动保存。/ JSON is saved automatically.\n' \
                "${S[prompt_tokens]}" "${S[gen_tokens]}" "${S[depth_mode]}" "${S[depth_tokens]}" "${S[reps]}"
        fi
    fi
}

add_arg() { [[ -z ${S[$2]} ]] || COMMAND+=("$1" "${S[$2]}"); return 0; }
setting() { COMMAND+=(--set "$1=$2"); }
build_command() {
    local key entry value flag; local -a entries
    [[ ${S[launch_mode]} =~ ^(run|serve|bench)$ && ${S[setup_mode]} =~ ^(conservative|aggressive|manual)$ ]] || { fail 'Invalid mode or profile.'; return 1; }
    [[ -n ${S[model]} ]] || { fail 'A model is required.'; return 1; }
    if [[ ${S[mtp_enabled]} == true && ${S[launch_mode]} != bench ]]; then
        [[ -n ${S[mtp_model]} && ${S[mtp_verify_tokens]} =~ ^[234]$ && (${S[device]} == '' || ${S[device]} =~ ^Vulkan[0-9]+$) ]] || { fail 'MTP requires a head, Vulkan GPU and width 2-4.'; return 1; }
        if [[ ${S[launch_mode]} == serve ]]; then
            [[ ${S[server_parallel]} == 1 || ${S[server_parallel]} == 2 ]] || { fail 'MTP supports one or two configured slots.'; return 1; }
            [[ ${S[server_parallel]} != 2 ]] || S[mtp_verify_tokens]=4
        fi
    fi
    COMMAND=("$BINARY" "${S[launch_mode]}"); add_arg --dev device
    if [[ ${S[setup_mode]} == manual ]]; then add_arg --config config_path; add_arg --ubatch ubatch; add_arg --threads threads
    else
        setting device.auto_profile "${S[setup_mode]}"
        [[ ,${S[auto_overrides]}, != *,ubatch,* ]] || add_arg --ubatch ubatch
        for key in ram_budget vram_budget; do if [[ ,${S[auto_overrides]}, == *,$key,* && -n ${S[$key]} ]]; then setting "device.$key" "${S[$key]}"; fi; done
    fi
    [[ ${S[launch_mode]} == bench ]] || add_arg --ctx context
    if [[ ${S[kv_preset]} != auto ]]; then setting kv.type_k "${S[kv_type_k]}"; setting kv.type_v "${S[kv_type_v]}"; fi
    if [[ ${S[setup_mode]} == manual ]]; then
        if [[ ${S[configure_memory]} == true ]]; then
            for entry in 'vram_budget|device.vram_budget' 'vram_reserve|device.vram_reserve' 'expert_cache|paging.cache' 'ram_budget|device.ram_budget' 'pager_ring|paging.ring' 'pager_ring_slots|paging.ring_slots'; do
                key=${entry%%|*}; [[ -z ${S[$key]} ]] || setting "${entry#*|}" "${S[$key]}"
            done
            setting paging.host_dma "${S[host_dma]}"; setting kv.overflow "${S[kv_overflow]}"
            if [[ ${S[kv_overflow]} == true ]]; then for key in kv_overflow_vram_mb kv_overflow_reserve_mb; do [[ -z ${S[$key]} ]] || setting "kv.${key#kv_}" "${S[$key]}"; done; fi
        fi
        case ${S[submit_mode]} in disabled) setting device.submit_dispatches 0 ;; fixed) setting device.submit_dispatches "${S[submit_cap]}" ;; esac
        if [[ ${S[configure_diagnostics]} == true ]]; then
            setting paging.stats "${S[pager_stats]}"; setting prof.pager_profile "${S[pager_profile]}"; setting prof.stages "${S[stage_profile]}"; setting prof.vram "${S[vram_profile]}"
        fi
    fi
    if [[ ${S[cpu_miss_enabled]} == true ]]; then
        valid_uint "${S[cpu_miss_cores]}" 1 && valid_uint "${S[cpu_miss_max]}" 1 3 || { fail 'Invalid CPU-miss cores or miss limit.'; return 1; }
        setting kernels.vulkan.cpu_miss_threads "${S[cpu_miss_cores]}"; setting kernels.vulkan.cpu_miss_max "${S[cpu_miss_max]}"
        for key in push host_result token_park; do setting "kernels.vulkan.cpu_miss_$key" true; done
        setting kernels.vulkan.cpu_miss_spin 262144
    else setting kernels.vulkan.cpu_miss_threads 0; fi
    if [[ ${S[setup_mode]} == manual ]]; then
        IFS=';' read -r -a entries <<< "${S[custom_sets]}"
        for entry in "${entries[@]}"; do
            trim "$entry"; entry=$REPLY; [[ -n $entry ]] || continue
            [[ $entry == *=* ]] || { fail 'Extra --set requires key=value.'; return 1; }
            trim "${entry%%=*}"; key=$REPLY; trim "${entry#*=}"; value=$REPLY
            [[ -n $key && $key != serve.api_key ]] || { fail 'Use the dedicated API-key prompt.'; return 1; }
            if [[ $key == kernels.vulkan.cpu_miss_* ]]; then
                local duplicate=false
                for flag in "${COMMAND[@]}"; do [[ $flag != "$key="* ]] || duplicate=true; done
                [[ $duplicate == false ]] || continue
            fi
            setting "$key" "$value"
        done
    fi
    if [[ ${S[launch_mode]} == bench ]]; then
        if [[ ${S[bench_kind]} =~ ^(quick|standard|full)$ ]]; then
            COMMAND+=(--suite "${S[bench_kind]}" --parallel "${S[bench_parallel]}" --result-dir "$ROOT/benchmark-database")
        else
            if [[ ${S[manual_bench_kind]} == mixed ]]; then COMMAND+=(--pg "${S[prompt_tokens]},${S[gen_tokens]}"); else COMMAND+=(-p "${S[prompt_tokens]}" -n "${S[gen_tokens]}"); fi
            case ${S[depth_mode]} in real) COMMAND+=(-d "${S[depth_tokens]}") ;; synthetic) COMMAND+=(--synthetic-depth "${S[depth_tokens]}") ;; esac
            COMMAND+=(-r "${S[reps]}" --json)
        fi
    else
        setting spec.mtp "${S[mtp_enabled]}"
        if [[ ${S[mtp_enabled]} == true ]]; then setting spec.draft "${S[mtp_model]}"; setting spec.k "${S[mtp_verify_tokens]}"; fi
        case ${S[think_mode]} in think|no-think) COMMAND+=("--${S[think_mode]}") ;; esac
        if [[ ${S[think_mode]} != no-think && ${S[reasoning_effort]} != default ]]; then add_arg --reasoning-effort reasoning_effort; fi
        add_arg --max-new max_new
        if [[ ${S[mtp_enabled]} == true ]]; then COMMAND+=(--temp 0)
        elif [[ ${S[configure_sampling]} == true ]]; then add_arg --temp temperature; add_arg --top-k top_k; add_arg --top-p top_p; add_arg --seed seed; fi
    fi
    if [[ ${S[launch_mode]} == serve ]]; then
        valid_address "${S[server_addr]}" || { fail 'Invalid listen address.'; return 1; }
        valid_uint "${S[server_parallel]}" 1 || { fail 'Invalid concurrent slot count.'; return 1; }
        [[ ${S[server_auth]} == true ]] || setting serve.api_key ''
        add_arg --addr server_addr; add_arg --parallel server_parallel
        if [[ ${S[server_session_cache]} == true ]] && ! serial_mtp; then
            for key in session_cache_dir session_idle_secs session_cache_max session_cache_ttl_hours; do setting "kv.$key" "${S[$key]}"; done
        else setting kv.session_cache_dir ''; fi
        [[ ${S[server_vision]} != true ]] || { [[ -n ${S[vision_projector]} ]] || return 1; add_arg --mmproj vision_projector; }
        if [[ ${S[server_embedding]} == true ]]; then [[ -n ${S[embedding_model]} ]] || return 1; add_arg --embedding-model embedding_model; add_arg --embedding-idle-timeout embedding_idle_timeout; fi
    fi
    COMMAND+=("${S[model]}")
}

version_valid() { [[ $1 =~ ^(0|[1-9][0-9]{0,5})\.(0|[1-9][0-9]{0,5})\.(0|[1-9][0-9]{0,5})$ ]]; }
version_newer() {
    version_valid "$1" && version_valid "$2" || return 1
    local -a a b; local i
    IFS=. read -r -a a <<< "$1"; IFS=. read -r -a b <<< "$2"
    for i in 0 1 2; do ((a[i]>b[i])) && return 0; ((a[i]<b[i])) && return 1; done
    return 1
}
managed() {
    [[ -n $1 && $1 != /* && $1 != *\\* && ! $1 =~ (^|/)(\.|\.\.|)(/|$) && ! $1 =~ [[:cntrl:]] ]] || return 1
    case $1 in
        infr|Start-INFR-Wizard-Linux.sh|README.md|README_EN.md|CHANGELOG.md|infr.example.toml|LICENSE|LICENSE-MIT|NOTICE|install-manifest.json) return 0 ;;
        image-codecs/libheif.so.1|image-codecs/libde265.so.0|image-codecs/libdav1d.so.7|image-codecs/libheif-LICENSE.txt|image-codecs/libde265-LICENSE.txt|image-codecs/dav1d-LICENSE.txt|image-codecs/SOURCES.txt) return 0 ;;
        documentation/*) [[ ${1,,} =~ \.(md|png|jpg|svg|webp)$ ]] ;;
        *) return 1 ;;
    esac
}
safe_target() {
    local relative=$2 current=$1 part index=0; local -a parts
    managed "$relative" || return 1
    IFS=/ read -r -a parts <<< "$relative"
    for part in "${parts[@]}"; do
        current+="/$part"; [[ ! -L $current ]] || return 1; ((index+=1))
        if ((index<${#parts[@]})); then [[ ! -e $current || -d $current ]] || return 1; fi
    done
    [[ ! -e $current || -f $current ]] || return 1
    REPLY=$current
}
validate_package() {
    local base=$1 version=$2 index=0 name size hash file expected path actual
    local -A seen=()
    FILES=()
    json_load "$base/install-manifest.json" || return 1
    [[ ${J['["schema_version"]']-} == 1 && ${J['["updater_protocol"]']-} == 1 && ${J['["product"]']-} == '"moe4all-engine"' && ${J['["platform"]']-} == '"Linux-x86_64"' && ${J['["version"]']-} == "\"$version\"" ]] || return 1
    while true; do
        path="[\"files\",$index,\"path\"]"; [[ -v 'J[$path]' ]] || break
        json_get "$path" || return 1; name=$REPLY
        [[ $name != install-manifest.json && ! -v 'seen[$name]' ]] || return 1
        safe_target "$base" "$name" || return 1; file=$REPLY
        [[ -f $file ]] || return 1
        path="[\"files\",$index,\"size\"]"; size=${J[$path]-}
        [[ $size =~ ^[0-9]{1,10}$ ]] && ((size<=1073741824)) || return 1
        json_get "[\"files\",$index,\"sha256\"]" || return 1; hash=$REPLY
        [[ $hash =~ ^[0-9a-fA-F]{64}$ && $(wc -c < "$file") == "$size" ]] || return 1
        actual=$(sha256sum -- "$file"); actual=${actual%% *}
        [[ $actual == "${hash,,}" ]] || return 1
        seen[$name]=1; FILES+=("$name"); ((index+=1))
        ((index<=6000)) || return 1
    done
    [[ -v 'seen[infr]' && -v 'seen[Start-INFR-Wizard-Linux.sh]' ]] || return 1
    expected=$((${#FILES[@]}+1)); actual=0
    while IFS= read -r -d '' file; do
        name=${file#"$base/"}; [[ $name == install-manifest.json || -v 'seen[$name]' ]] || return 1
        ((actual+=1))
    done < <(find "$base" -type f -print0)
    [[ $actual == "$expected" && -z $(find "$base" ! -type f ! -type d -print -quit) ]]
}
extract_package() {
    local archive=$1 destination=$2 version=$3 prefix="MoE4All-Linux-x86_64-v$3" names verbose name relative count=0
    local -A seen=()
    names=$(tar --quoting-style=escape -tzf "$archive") || return 1
    verbose=$(tar --quoting-style=escape -tvzf "$archive") || return 1
    # Refuse links/devices before extraction; cap declared payload and member count.
    printf '%s\n' "$verbose" | awk 'substr($0,1,1)!="-"&&substr($0,1,1)!="d"{exit 1}{n++;bytes+=$3}END{if(n>6000||bytes>1073741824)exit 1}' || return 1
    while IFS= read -r name; do
        [[ $name != *\\* && ! $name =~ (^|/)(\.|\.\.)(/|$) && ! $name =~ [[:cntrl:]] && ! -v 'seen[$name]' ]] || return 1
        seen[$name]=1; ((count+=1)); ((count<=6000)) || return 1
        [[ $name == "$prefix" || $name == "$prefix/" ]] && continue
        [[ $name == "$prefix/"* ]] || return 1
        relative=${name#"$prefix/"}
        if [[ $relative == */ ]]; then
            [[ $relative == documentation/ || $relative == documentation/*/ || $relative == image-codecs/ ]] || return 1
        else managed "$relative" || return 1; fi
    done <<< "$names"
    mkdir -p -- "$destination"
    tar --no-same-owner --no-same-permissions --keep-old-files --delay-directory-restore --strip-components=1 -xzf "$archive" -C "$destination" || return 1
    validate_package "$destination" "$version" || return 1
    chmod 755 "$destination/infr" "$destination/Start-INFR-Wizard-Linux.sh"
}
engine_running() {
    local path
    for path in /proc/[0-9]*/exe; do [[ ! $path -ef $1 ]] || return 0; done
    return 1
}
apply_update() (
    local root=$1 stage=$2 version=$3 old_version name target backup success=false actual
    local -a installed=() files
    [[ ! -e $root/.git && -f $root/install-manifest.json && ! -L $root/install-manifest.json ]] || { fail 'Only an extracted managed package can be updated.'; exit 1; }
    json_load "$root/install-manifest.json" || exit 1
    [[ ${J['["product"]']-} == '"moe4all-engine"' && ${J['["platform"]']-} == '"Linux-x86_64"' && ${J['["updater_protocol"]']-} == 1 ]] || exit 1
    json_get '["version"]' || exit 1; old_version=$REPLY
    version_newer "$version" "$old_version" || { fail 'Refusing a downgrade or same-version replacement.'; exit 1; }
    validate_package "$stage" "$version" || { fail 'Invalid staged package; no files changed.'; exit 1; }
    bash -n "$stage/Start-INFR-Wizard-Linux.sh" || { fail 'Invalid staged launcher; no files changed.'; exit 1; }
    files=("${FILES[@]}" install-manifest.json)
    for name in "${files[@]}"; do safe_target "$root" "$name" || { fail 'Unsafe install destination; no files changed.'; exit 1; }; done
    if engine_running "$root/infr"; then fail 'Stop the running engine before updating; no files changed.'; exit 1; fi
    backup=$(mktemp -d "$root/.update-backup.XXXXXX") || exit 1
    rollback() {
        local status=$? name target errors=false
        trap - EXIT
        if [[ $success == false ]]; then
            for name in "${installed[@]}"; do
                target="$root/$name"
                if [[ -f $backup/$name ]]; then cp -p -- "$backup/$name" "$target" || errors=true
                else rm -f -- "$target" || errors=true; fi
            done
            if [[ $errors == true ]]; then printf 'Rollback incomplete; keep backup: %s\n' "$backup" >&2
            else rm -rf -- "$backup"; printf 'Update failed; previous installation restored.\n' >&2; fi
            exit 1
        fi
        rm -rf -- "$backup"; exit "$status"
    }
    trap rollback EXIT
    for name in "${files[@]}"; do
        target="$root/$name"
        if [[ -f $target ]]; then mkdir -p -- "$(dirname -- "$backup/$name")" || exit 1
            cp -p -- "$target" "$backup/$name" || exit 1
        fi
    done
    for name in "${files[@]}"; do
        target="$root/$name"; mkdir -p -- "$(dirname -- "$target")" || exit 1
        mv -f -- "$stage/$name" "$target" || exit 1; installed+=("$name")
    done
    actual=$("$root/infr" --version) || exit 1
    [[ $actual == "infr $version" ]] || exit 1
    bash -n "$root/Start-INFR-Wizard-Linux.sh" || exit 1
    success=true
)
fetch_url() {
    local url=$1 output=$2 maximum=$3 seconds=${4-600} progress=${5-false}
    local -a curl_args=(-fL --proto '=https' --proto-redir '=https' --connect-timeout 5 --max-time "$seconds" --max-filesize "$maximum" -H 'Accept: application/vnd.github+json' -H 'User-Agent: MoE4All-Linux-Wizard' -o "$output")
    [[ $url == https://api.github.com/repos/Headmaster218/MoE4All/releases\?per_page=100 || $url =~ ^https://github\.com/Headmaster218/MoE4All/releases/download/release-[0-9]+\.[0-9]+\.[0-9]+/MoE4All-Linux-x86_64-v[0-9]+\.[0-9]+\.[0-9]+\.tar\.gz(\.sha256)?$ ]] || return 1
    if [[ $progress == true && -t 2 ]]; then curl_args+=(--show-error --progress-bar)
    else curl_args+=(-sS); fi
    curl "${curl_args[@]}" "$url" || return 1
    [[ $(wc -c < "$output") -le $maximum ]]
}
select_release() {
    local i=0 j tag version name url path archive checksum
    RELEASE_VERSION=''; RELEASE_ARCHIVE=''; RELEASE_CHECKSUM=''
    while true; do
        path="[$i,\"tag_name\"]"; [[ -v 'J[$path]' ]] || break
        json_get "$path" || return 1; tag=$REPLY
        if [[ ${J["[$i,\"draft\"]"]-} == false && ${J["[$i,\"prerelease\"]"]-} == false && $tag == release-* ]]; then
            version=${tag#release-}
            if version_valid "$version" && { [[ -z $RELEASE_VERSION ]] || version_newer "$version" "$RELEASE_VERSION"; }; then
                RELEASE_VERSION=$version; archive=''; checksum=''; j=0; name="MoE4All-Linux-x86_64-v$version.tar.gz"
                while true; do
                    path="[$i,\"assets\",$j,\"name\"]"; [[ -v 'J[$path]' ]] || break
                    json_get "$path" || return 1; path=$REPLY
                    json_get "[$i,\"assets\",$j,\"browser_download_url\"]" || return 1; url=$REPLY
                    [[ $path != "$name" ]] || archive=$url; [[ $path != "$name.sha256" ]] || checksum=$url
                    ((j+=1))
                done
                RELEASE_ARCHIVE=$archive; RELEASE_CHECKSUM=$checksum
            fi
        fi
        ((i+=1))
    done
    [[ -n $RELEASE_VERSION ]]
}
download_update() (
    local root=$1 relaunch=${2-true} version=$RELEASE_VERSION temp name="MoE4All-Linux-x86_64-v$RELEASE_VERSION.tar.gz" hash filename extra actual
    [[ $relaunch == true || $relaunch == false ]] || { fail 'Invalid updater relaunch mode.'; exit 1; }
    [[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || { fail 'Updating supports Linux x86_64 only.'; exit 1; }
    need flock || exit 1; need tar || exit 1; need sha256sum || exit 1
    [[ ! -L $root/.moe4all-update.lock ]] || { fail 'Refusing a symlinked update lock.'; exit 1; }
    exec 9>"$root/.moe4all-update.lock"; flock -n 9 || { fail 'Another update is in progress.'; exit 1; }
    temp=$(mktemp -d "$root/.update-stage.XXXXXX") || exit 1
    trap 'rm -rf -- "$temp"' EXIT
    fetch_url "$RELEASE_CHECKSUM" "$temp/checksum" 4096 || exit 1
    read -r hash filename extra < "$temp/checksum" || exit 1
    [[ $hash =~ ^[0-9a-fA-F]{64}$ && ${filename#\*} == "$name" && -z $extra ]] || { fail 'Malformed release checksum.'; exit 1; }
    fetch_url "$RELEASE_ARCHIVE" "$temp/$name" 1073741824 600 true || exit 1
    actual=$(sha256sum -- "$temp/$name"); actual=${actual%% *}
    [[ $actual == "${hash,,}" ]] || { fail 'Release checksum mismatch; no files changed.'; exit 1; }
    extract_package "$temp/$name" "$temp/files" "$version" || { fail 'Invalid release archive; no files changed.'; exit 1; }
    apply_update "$root" "$temp/files" "$version" || exit 1
    if [[ $relaunch == true ]]; then
        printf '更新完成，正在重新启动新版向导…… / Update complete. Restarting the updated launcher...\n'
        trap - EXIT
        rm -rf -- "$temp"
        flock -u 9
        exec 9>&-
        exec bash "$root/Start-INFR-Wizard-Linux.sh" --skip-update-check
    fi
    printf '更新完成。Update complete.\n'
)
check_updates() (
    local mode=$1 current temp
    if ! need curl; then printf '更新检查不可用，继续离线启动。Update check unavailable; continuing offline.\n'; return; fi
    current=$("$BINARY" --version) || return 0; current=${current##* }
    version_valid "$current" || return 0
    temp=$(mktemp -d); trap 'rm -rf -- "$temp"' EXIT
    if ! fetch_url 'https://api.github.com/repos/Headmaster218/MoE4All/releases?per_page=100' "$temp/releases.json" 8388608 10 || ! json_load "$temp/releases.json" || ! select_release; then
        printf '更新检查不可用，继续离线启动。Update check unavailable; continuing offline.\n'; return 0
    fi
    if ! version_newer "$RELEASE_VERSION" "$current"; then printf '更新检查 / Update check: v%s is current\n' "$current"; return 0; fi
    printf '发现新引擎版本 / New engine release: v%s\nhttps://github.com/Headmaster218/MoE4All/releases/tag/release-%s\n' "$RELEASE_VERSION" "$RELEASE_VERSION"
    if [[ -z $RELEASE_ARCHIVE || -z $RELEASE_CHECKSUM || ! -f $ROOT/install-manifest.json || $BINARY != "$ROOT/infr" || -e $ROOT/.git ]]; then
        printf '源码安装或没有兼容的 Linux 包：仅检查，未修改文件。Source install or no compatible Linux package: check only; no files changed.\n'
    elif [[ $mode != --update ]]; then printf '显式更新命令 / To update explicitly: bash ./Start-INFR-Wizard-Linux.sh --update\n'
    elif [[ $INTERACTIVE != true || $HEADLESS == true ]]; then printf '更新需要交互确认，未修改文件。Updating requires interactive confirmation; no files changed.\n'
    else yes '现在更新此 Linux 安装包？/ Update this Linux package now?' false; [[ $REPLY != true ]] || download_update "$ROOT"; fi
)

usage() {
    printf '%s\n' 'MoE4All Linux launch wizard (no Python or jq).' \
        'Usage: bash Start-INFR-Wizard-Linux.sh [options]' \
        '  --dry-run                 Print only; no prompts, save, network or launch' \
        '  --yes, -y                 Explicit headless launch (also permits no-auth LAN)' \
        '  --check-update            Check only, then exit' \
        '  --update                  Update a managed package after confirmation' \
        '  --skip-update-check       Skip normal startup update checks' \
        '  --mode run|serve|bench    --model FILE|DIRECTORY|HF_REFERENCE' \
        '  --profile conservative|aggressive|manual --dev VulkanN --ctx SIZE' \
        '  --ubatch N --threads N --config FILE --kv-k TYPE --kv-v TYPE' \
        '  --ram SIZE --vram SIZE --mtp HEAD --mtp-k 2|3|4' \
        '  --addr IP:PORT --parallel N --mmproj FILE --embedding FILE' \
        '  --max-new N --cpu-miss-cores N --cpu-miss-max 1|2|3' \
        '  --bench-p N --bench-n N --bench-d N' \
        '  --no-mtp --no-mmproj --no-embedding --no-api-key' \
        '  --root DIRECTORY          Resolve all relative paths against this root'
}
main() {
    local key flag value inherited=${INFR_API_KEY-} api_key='' reused=false before_benchmark='' result_path='' raw_benchmark=''
    local -a argv=("$@")
    ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
    DRY_RUN=false; HEADLESS=false; SKIP_UPDATE=false; UPDATE_MODE=''; INTERACTIVE=false; NO_API_KEY=false
    declare -A disables=()
    for ((key=0;key<${#argv[@]};key++)); do
        case ${argv[key]} in --help|-h) usage; return ;; esac
        if [[ ${argv[key]} == --root ]]; then
            ((key+1<${#argv[@]})) || { fail '--root requires a directory'; return 1; }
            ROOT=$(realpath -m -- "${argv[key+1]}")
        fi
    done
    [[ -d $ROOT ]] || { fail 'Installation root does not exist.'; return 1; }
    CONFIG_DIR="$ROOT/gui-data"; STATE="$CONFIG_DIR/wizard-state.json"
    LEGACY_CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/infr"; LEGACY_STATE="$LEGACY_CONFIG_DIR/wizard-state.json"
    need awk; defaults; load_state
    while (($#)); do
        flag=$1; shift
        case $flag in
            --root) shift ;; --dry-run) DRY_RUN=true ;; --yes|-y) HEADLESS=true ;; --skip-update-check) SKIP_UPDATE=true ;;
            --check-update|--update) [[ -z $UPDATE_MODE ]] || { fail 'Choose only one update mode.'; return 1; }; UPDATE_MODE=$flag ;;
            --no-api-key) NO_API_KEY=true ;; --no-mtp) disables[mtp_model]=mtp_enabled ;; --no-mmproj) disables[vision_projector]=server_vision ;; --no-embedding) disables[embedding_model]=server_embedding ;;
            *)
                case $flag in
                    --mode) key=launch_mode ;; --model) key=model ;; --profile) key=setup_mode ;; --dev) key=device ;; --ctx) key=context ;;
                    --ubatch|-u) key=ubatch ;; --threads) key=threads ;; --config) key=config_path ;; --kv-k) key=kv_type_k ;; --kv-v) key=kv_type_v ;;
                    --ram) key=ram_budget ;; --vram) key=vram_budget ;; --mtp) key=mtp_model ;; --mtp-k) key=mtp_verify_tokens ;;
                    --addr) key=server_addr ;; --parallel) key=server_parallel ;; --mmproj) key=vision_projector ;; --embedding) key=embedding_model ;;
                    --max-new) key=max_new ;; --cpu-miss-cores) key=cpu_miss_cores ;; --cpu-miss-max) key=cpu_miss_max ;;
                    --bench-p) key=prompt_tokens ;; --bench-n) key=gen_tokens ;; --bench-d) key=depth_tokens ;;
                    *) fail "Unknown option: $flag"; return 1 ;;
                esac
                (($#)) || { fail "$flag requires a value"; return 1; }; S[$key]=$1; OVERRIDES[$key]=1; shift ;;
        esac
    done
    for key in ubatch ram_budget vram_budget; do [[ ! -v 'OVERRIDES[$key]' ]] || S[auto_overrides]+=",$key,"; done
    # The release wizard no longer exposes MTP. Ignore stale saved MTP state,
    # while preserving the explicit --mtp CLI for advanced/manual use.
    if [[ ! -v 'OVERRIDES[mtp_model]' ]]; then S[mtp_enabled]=false; S[mtp_model]=''; S[mtp_verify_tokens]=4; fi
    for key in mtp_model vision_projector embedding_model; do
        case $key in mtp_model) value=mtp_enabled ;; vision_projector) value=server_vision ;; embedding_model) value=server_embedding ;; esac
        if [[ -v 'OVERRIDES[$key]' ]]; then S[$value]=false; [[ -z ${S[$key]} ]] || S[$value]=true; fi
    done
    for key in "${!disables[@]}"; do S[$key]=''; S[${disables[$key]}]=false; done
    [[ ! -v 'OVERRIDES[ram_budget]' && ! -v 'OVERRIDES[vram_budget]' ]] || S[configure_memory]=true
    [[ ! -v 'OVERRIDES[kv_type_k]' && ! -v 'OVERRIDES[kv_type_v]' ]] || S[kv_preset]=custom
    if [[ -v 'OVERRIDES[cpu_miss_cores]' ]]; then S[cpu_miss_enabled]=false; [[ ${S[cpu_miss_cores]} == '' || ${S[cpu_miss_cores]} == 0 ]] || S[cpu_miss_enabled]=true; fi
    [[ ! -v 'OVERRIDES[depth_tokens]' ]] || S[depth_mode]=real
    if [[ -v 'OVERRIDES[prompt_tokens]' || -v 'OVERRIDES[gen_tokens]' ]]; then S[bench_kind]=manual; S[manual_bench_kind]=custom; fi
    if [[ $NO_API_KEY == true ]]; then S[server_auth]=false; elif [[ -n $inherited ]]; then S[server_auth]=true; fi
    BINARY=''
    for value in "$ROOT/infr" "$ROOT/target/release/infr" "$(command -v infr || true)"; do
        if [[ -f $value && -x $value ]]; then BINARY=$(realpath -- "$value"); break; fi
    done
    [[ -n $BINARY ]] || { fail 'infr not found beside the launcher or under target/release; build infr-cli first.'; return 1; }
    [[ ! -t 0 || $DRY_RUN == true ]] || INTERACTIVE=true
    PENDING_SUBMISSION_PROMPTED=false
    if [[ -n $UPDATE_MODE ]]; then check_updates "$UPDATE_MODE"; return; fi
    if [[ $INTERACTIVE == true && $SKIP_UPDATE == false && ! ${MOE4ALL_NO_UPDATE_CHECK-} =~ ^(1|true|yes|on)$ ]]; then check_updates --check-update; fi
    if [[ $INTERACTIVE == true && $HEADLESS == false ]]; then
        retry_pending_submissions
        value=$("$BINARY" --version 2>/dev/null || true); value=${value##* }
        printf 'MoE4All v%s Linux 启动向导 / Launch Wizard\n上次设置会作为默认值；直接回车即可复用，输入 - 可清空。\nPress Enter to reuse the previous/default value; enter - to clear it.\n' "$value"
        if [[ -n ${S[model]} && ${#OVERRIDES[@]} == 0 && ${#disables[@]} == 0 ]]; then
            absolute "${S[model]}"; if [[ -f $REPLY ]]; then yes '是否按上次设置直接启动？/ Start with the previous settings?' true; reused=$REPLY; fi
        fi
        [[ $reused == true ]] || configure
        [[ $NO_API_KEY == false ]] || S[server_auth]=false
    fi
    if [[ ${S[launch_mode]} == report ]]; then
        complete_error_report
        S[launch_mode]=${REPORT_RETURN_MODE:-run}
        save_state
        return 0
    fi
    if [[ ${S[launch_mode]} == serve && ${S[server_auth]} == true ]]; then
        api_key=$inherited
        if [[ $INTERACTIVE == true && $HEADLESS == false ]]; then
            printf 'API key（隐藏输入；回车复用环境变量）/ API key (hidden; Enter reuses inherited key): '; IFS= read -rs value || return 1; printf '\n'; [[ -z $value ]] || api_key=$value
        fi
        [[ -n $api_key ]] || { fail 'API-key authentication requires INFR_API_KEY or hidden input.'; return 1; }
    fi
    for key in model mtp_model vision_projector embedding_model config_path session_cache_dir; do
        [[ -n ${S[$key]} ]] || continue
        if [[ $key == model ]] && hub_reference "${S[$key]}"; then continue; fi
        absolute "${S[$key]}"; S[$key]=$REPLY
    done
    build_command
    if [[ $INTERACTIVE == true && $DRY_RUN == false && $HEADLESS == false ]]; then retry_pending_submissions; fi
    printf '\n最终命令 / Final command\n  '; printf '%q ' "${COMMAND[@]}"; printf '\n'
    if [[ ${S[launch_mode]} == serve && -z $api_key ]] && ! loopback "${S[server_addr]}"; then
        printf '警告：当前网络监听地址未启用 API key 鉴权。Warning: network listen address without API-key authentication.\n' >&2
        if [[ $DRY_RUN == false && $HEADLESS == false ]]; then
            [[ $INTERACTIVE == true ]] || { fail 'Unauthenticated network serving requires explicit --yes.'; return 1; }
            yes '仍然继续？/ Continue without authentication?' false; [[ $REPLY == true ]] || return 0
        fi
    fi
    [[ $DRY_RUN == false ]] || return 0
    if [[ $HEADLESS == false && $reused == false ]]; then
        [[ $INTERACTIVE == true ]] || { fail 'Launch confirmation requires input or --yes.'; return 1; }
        if [[ ${S[launch_mode]} == serve ]]; then yes '现在启动服务器？/ Start the server now?' true
        else yes '现在启动？/ Start now?' true; fi
        [[ $REPLY == true ]] || return 0
    fi
    remember_model; save_state; printf '设置已保存 / Settings saved: %s\n' "$STATE"
    if [[ -n $api_key ]]; then export INFR_API_KEY=$api_key; else unset INFR_API_KEY; fi
    export INFR_DIAGNOSTIC_DIR="$ROOT/diagnostics"
    cd -- "$ROOT"
    if [[ $INTERACTIVE == true && $HEADLESS == false ]]; then
        local before_report after_report exit_code
        newest_error_report; before_report=$REPLY
        if [[ ${S[launch_mode]} == bench ]]; then newest_benchmark_report; before_benchmark=$REPLY; fi
        if [[ ${S[launch_mode]} == bench && ${S[bench_kind]} == manual ]]; then
            mkdir -p -- "$CONFIG_DIR"
            raw_benchmark=$(mktemp "$CONFIG_DIR/.manual-benchmark.XXXXXX")
            if "${COMMAND[@]}" | tee "$raw_benchmark"; then
                exit_code=0
                if pack_manual_benchmark "$raw_benchmark"; then result_path=$REPLY; fi
            else
                exit_code=$?
            fi
            rm -f -- "$raw_benchmark"
        else
            if "${COMMAND[@]}"; then exit_code=0; else exit_code=$?; fi
            if [[ $exit_code == 0 && ${S[launch_mode]} == bench ]]; then
                newest_benchmark_report
                [[ -z $REPLY || $REPLY == "$before_benchmark" ]] || result_path=$REPLY
            fi
        fi
        if [[ -n $result_path ]]; then offer_benchmark_upload "$result_path" || true; fi
        newest_error_report; after_report=$REPLY
        if ((exit_code != 0)) && [[ -n $after_report && $after_report != "$before_report" ]]; then complete_error_report "$after_report"; save_state; fi
        return "$exit_code"
    else
        exec "${COMMAND[@]}"
    fi
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    trap 'launcher_exit $?' EXIT
    main "$@"
    trap - EXIT
fi
