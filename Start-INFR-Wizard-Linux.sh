#!/usr/bin/env bash
# Runtime: Bash 4+, awk and standard Linux utilities; no Python or jq.
set -euo pipefail

fail() { printf '%s\n' "$*" >&2; return 1; }
need() { command -v "$1" >/dev/null || fail "Required system tool not found: $1"; }
trim() { REPLY=$1; REPLY="${REPLY#"${REPLY%%[![:space:]]*}"}"; REPLY="${REPLY%"${REPLY##*[![:space:]]}"}"; }

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

# vim:fdm=marker:
MOE4ALL_JSON_AWK

declare -A S=() DEFAULTS=() J=() OVERRIDES=()
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
        [launch_mode]=run [setup_mode]=conservative [model]='' [device]='' [context]=''
        [ubatch]='' [threads]='' [config_path]='' [kv_preset]=q8 [kv_type_k]=q8_0 [kv_type_v]=q8_0
        [configure_memory]=false [vram_budget]='' [vram_reserve]='' [expert_cache]='' [ram_budget]=''
        [host_dma]=true [pager_ring]='' [pager_ring_slots]='' [kv_overflow]=false [kv_overflow_vram_mb]='' [kv_overflow_reserve_mb]=''
        [submit_mode]=auto [submit_cap]=64 [configure_diagnostics]=false [pager_stats]=false [pager_profile]=false [stage_profile]=false [vram_profile]=false
        [think_mode]=default [configure_thinking]=false [reasoning_effort]=default [max_new]=65536
        [configure_sampling]=false [temperature]='' [top_k]='' [top_p]='' [seed]=''
        [server_addr]=127.0.0.1:8080 [server_parallel]=1 [server_auth]=false [server_session_cache]=false
        [session_cache_dir]="$ROOT/kv-sessions" [session_idle_secs]=120 [session_cache_max]=5GiB [session_cache_ttl_hours]=24
        [server_vision]=false [vision_projector]='' [server_embedding]=false [embedding_model]='' [embedding_idle_timeout]=300
        [mtp_enabled]=false [mtp_model]='' [mtp_verify_tokens]=4 [cpu_miss_enabled]=false [cpu_miss_max]=1 [cpu_miss_cores]=''
        [bench_kind]=decode [prompt_tokens]=1024 [gen_tokens]=128 [depth_mode]=none [depth_tokens]=0 [reps]=1 [json_output]=false
        [custom_sets]='' [auto_overrides]=''
    )
    DEFAULTS=(); for key in "${!S[@]}"; do DEFAULTS[$key]=${S[$key]}; done
}
load_state() {
    local key raw value path
    if [[ -f $STATE ]]; then
        json_load "$STATE" || { fail "Invalid saved settings: $STATE"; return 1; }
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
    elif [[ -f $CONFIG_DIR/wizard.conf ]]; then
        local -A names=([MODE]=launch_mode [MODEL]=model [PROFILE]=setup_mode [CTX]=context [UBATCH]=ubatch [KV_K]=kv_type_k [KV_V]=kv_type_v
            [RAM]=ram_budget [VRAM]=vram_budget [MTP]=mtp_model [MTP_K]=mtp_verify_tokens [ADDR]=server_addr [PARALLEL]=server_parallel [MMPROJ]=vision_projector [EMBEDDING]=embedding_model)
        while IFS= read -r value; do
            [[ $value == *=* ]] || continue; key=${value%%=*}; [[ ! -v 'names[$key]' ]] || S[${names[$key]}]=${value#*=}
        done < "$CONFIG_DIR/wizard.conf"
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
        IFS= read -r value || { fail 'Launch cancelled: input ended.'; return 1; }; trim "$value"; value=$REPLY
        if [[ $value == - ]]; then value=''; elif [[ -z $value ]]; then value=${2-}; fi
        if [[ -n $value || ${3-false} == false ]]; then REPLY=$value; return; fi
        printf 'A value is required.\n'
    done
}
ask() { prompt "$2" "${S[$1]}" "${3-false}"; S[$1]=$REPLY; }
yes() {
    local value
    while true; do
        if [[ ${2-false} == true ]]; then printf '%s [Y/n]: ' "$1"; else printf '%s [y/N]: ' "$1"; fi
        IFS= read -r value || { fail 'Launch cancelled: input ended.'; return 1; }; trim "$value"; value=${REPLY,,}
        case $value in '') REPLY=${2-false}; return ;; y|yes) REPLY=true; return ;; n|no) REPLY=false; return ;; esac
        printf 'Enter y or n.\n'
    done
}
ask_yes() { yes "$2" "${S[$1]}"; S[$1]=$REPLY; }
valid_uint() { [[ $1 =~ ^[0-9]{1,9}$ ]] && ((10#$1 >= ${2-0} && 10#$1 <= ${3-999999999})); }
integer() {
    while true; do
        prompt "$2" "${S[$1]}"
        if [[ -z $REPLY && ${5-false} == true ]] || valid_uint "$REPLY" "${3-0}" "${4-999999999}"; then S[$1]=$REPLY; return; fi
        printf 'Integer outside the allowed range.\n'
    done
}
choice() {
    local key=$1 label=$2 selected i entry; shift 2
    local -a values=() labels=()
    for entry in "$@"; do values+=("${entry%%|*}"); labels+=("${entry#*|}"); done
    printf '\n%s\n' "$label"
    for ((i=0;i<${#values[@]};i++)); do selected=' '; [[ ${S[$key]} != "${values[i]}" ]] || selected='*'; printf ' %s[%d] %s\n' "$selected" "$((i+1))" "${labels[i]}"; done
    while true; do
        printf 'Select (Enter = previous/default): '
        IFS= read -r selected || { fail 'Launch cancelled: input ended.'; return 1; }; trim "$selected"; selected=$REPLY
        if [[ -z $selected ]]; then
            for entry in "${values[@]}"; do [[ ${S[$key]} != "$entry" ]] || return 0; done
        elif valid_uint "$selected" 1 "${#values[@]}"; then
            S[$key]=${values[10#$selected-1]}; return 0
        fi
        for ((i=0;i<${#values[@]};i++)); do
            if [[ $selected == "${values[i]}" ]]; then S[$key]=${values[i]}; return; fi
        done
        printf 'Choose a listed option.\n'
    done
}

absolute() {
    local value=$1
    [[ $value != '~' ]] || value=$HOME
    [[ $value != '~/'* ]] || value="$HOME/${value:2}"
    [[ $value == /* ]] || value="$ROOT/$value"
    REPLY=$(realpath -m -- "$value")
}
recommendations() {
    case $1 in
        llm) printf '%s\n' 'Qwen3.6 35B APEX-I-Balanced' 'https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-GGUF/resolve/main/Qwen3.6-35B-A3B-APEX-I-Balanced.gguf?download=true' 'Qwen3.8 Flash-Next AD-4.27bpw (33 shards)' 'https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64' ;;
        mtp) printf '%s\n' 'Flash-Next shared Q4_K_M MTP' 'https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf?download=true' ;;
        vision) printf '%s\n' 'Flash-Next F16 mmproj' 'https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/resolve/main/mmproj-Qwen3.8-Flash-Next-F16.gguf?download=true' ;;
        embedding) printf '%s\n' 'Qwen3-Embedding 0.6B' 'https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/tree/main' ;;
    esac
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
    local -a nearby=("${CANDIDATES[@]}") remembered=("$saved")
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
        printf '\n%s model; R = official recommended models\n' "${kind^^}"
        for ((i=0;i<${#nearby[@]};i++)); do printf ' [%d] %s\n' "$((i+1))" "${nearby[i]}"; done
        [[ -n $saved ]] || recommendations "$kind"
        value=$saved; [[ -n $value || $kind != llm || ${#nearby[@]} == 0 ]] || value=${nearby[0]}
        prompt 'GGUF file/directory, number, or R' "$value" true; value=$REPLY
        if [[ ${value,,} == r || ${value,,} == recommended ]]; then recommendations "$kind"; continue; fi
        if valid_uint "$value" 1 "${#nearby[@]}"; then value=${nearby[10#$value-1]}; fi
        if [[ $value == \"*\" ]]; then value=${value:1:${#value}-2}; fi
        absolute "$value"; path=$REPLY
        if [[ -f $path && ${path,,} == *.gguf ]]; then S[$key]=$path; return; fi
        if [[ -d $path ]]; then
            model_candidates "$path" "$kind"
            if ((${#CANDIDATES[@]}==1)); then S[$key]=${CANDIDATES[0]}; return; fi
            printf 'Directory has %d candidates; enter the exact file path.\n' "${#CANDIDATES[@]}"; continue
        fi
        if [[ $kind == llm ]] && hub_reference "$value"; then S[$key]=$value; return; fi
        printf 'GGUF file not found.\n'
    done
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
        if(template~/(^|[^a-zA-Z0-9_])reasoning_effort([^a-zA-Z0-9_]|$)/){if(arch=="qwen4exp")print "low medium xhigh";else print "low medium high max"}
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
    output=$("$BINARY" devices) || { fail 'Vulkan device enumeration failed; check the GPU driver.'; return 1; }
    while IFS= read -r line; do
        if [[ $line =~ ^[[:space:]]*(Vulkan[0-9]+):[[:space:]]+(.+)[[:space:]]+\[([^]]+)\](.*)$ ]]; then
            device=${BASH_REMATCH[1]}; name=${BASH_REMATCH[2]}; kind=${BASH_REMATCH[3]}; mark=${BASH_REMATCH[4]}
            [[ ${kind,,} != *cpu* ]] || continue
            options+=("$device|$device: $name [$kind]"); [[ $mark != *default* ]] || default=$device
        fi
    done <<< "$output"
    ((${#options[@]}>0)) || { fail 'No Vulkan GPU found; check the GPU driver.'; return 1; }
    [[ -n $default ]] || default=${options[0]%%|*}
    for entry in "${options[@]}"; do [[ ${S[device]} != "${entry%%|*}" ]] || found=true; done
    [[ $found == true ]] || S[device]=$default
    choice device '3. Runtime device' "${options[@]}"
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

configure() {
    local key value effort count previous
    choice launch_mode '1. Purpose' 'run|Terminal chat (recommended)' 'serve|OpenAI-compatible API' 'bench|Benchmark'
    model_path llm model
    if [[ ${S[launch_mode]} != bench ]]; then
        ask_yes mtp_enabled '2.2 Enable Qwen3.8 MTP?'
        if [[ ${S[mtp_enabled]} == true ]]; then
            model_path mtp mtp_model; choice mtp_verify_tokens 'MTP verification width' '4|4 (recommended)' '3|3' '2|2'
            printf 'Qwen3.8 Vulkan, greedy only. Non-greedy API requests use ordinary decode.\n'
        fi
    fi
    if [[ ${S[launch_mode]} == serve ]]; then
        if [[ -z ${S[vision_projector]} ]]; then model_candidates "${S[model]%/*}" vision; ((${#CANDIDATES[@]}!=1)) || S[vision_projector]=${CANDIDATES[0]}; fi
        ask_yes server_vision '2.3 Enable vision?'; [[ ${S[server_vision]} != true ]] || model_path vision vision_projector
        ask_yes server_embedding '2.4 Enable Embedding API?'
        if [[ ${S[server_embedding]} == true ]]; then model_path embedding embedding_model; integer embedding_idle_timeout 'Embedding weight idle seconds (0 = resident)'; fi
    fi
    choose_device
    choice setup_mode '4. Configuration' 'conservative|Automatic: conservative (recommended)' 'aggressive|Automatic: aggressive' 'manual|Manual'
    if [[ ${S[setup_mode]} == manual ]]; then
        integer ubatch 'Ubatch (blank = auto)' 1 999999999 true; integer threads 'CPU threads (blank = all)' 1 999999999 true
        while true; do ask config_path 'Config TOML (blank = default lookup)'; [[ -n ${S[config_path]} ]] || break; absolute "${S[config_path]}"; [[ ! -f $REPLY ]] || break; printf 'Config TOML not found.\n'; done
        choice kv_preset 'KV cache' 'auto|Engine default' 'q8|Q8_0 K + V' 'f16|F16 K + V' 'custom|Custom K / V'
        case ${S[kv_preset]} in custom) ask kv_type_k 'K dtype' true; ask kv_type_v 'V dtype' true ;; q8) S[kv_type_k]=q8_0; S[kv_type_v]=q8_0 ;; f16) S[kv_type_k]=f16; S[kv_type_v]=f16 ;; esac
        ask_yes configure_memory 'Configure memory and paging?'
        if [[ ${S[configure_memory]} == true ]]; then
            for key in vram_budget vram_reserve expert_cache ram_budget pager_ring; do ask "$key" "$key (blank = auto)"; done
            ask_yes host_dma 'Enable RAM-to-VRAM Host DMA?'; integer pager_ring_slots 'Pager ring slots' 2 999999999 true
            ask_yes kv_overflow 'Allow KV overflow to RAM?'
            if [[ ${S[kv_overflow]} == true ]]; then integer kv_overflow_vram_mb 'KV overflow VRAM MiB' 1 999999999 true; integer kv_overflow_reserve_mb 'KV overflow reserve MiB' 1 999999999 true; fi
        fi
        choice submit_mode 'Submit splitter' 'auto|Automatic feedback' 'disabled|Disabled / no-split' 'fixed|Fixed cap'
        [[ ${S[submit_mode]} != fixed ]] || integer submit_cap 'Dispatch cap' 1
        ask_yes configure_diagnostics 'Configure statistics / profilers?'
        if [[ ${S[configure_diagnostics]} == true ]]; then for key in pager_stats pager_profile stage_profile vram_profile; do ask_yes "$key" "$key"; done; fi
        ask custom_sets 'Extra --set (semicolon separated)'
    else S[kv_preset]=q8; S[kv_type_k]=q8_0; S[kv_type_v]=q8_0; fi
    S[auto_overrides]=''
    if [[ ${S[launch_mode]} != bench ]]; then
        ask_yes configure_thinking 'Configure default reasoning? (API may override)'
        previous=${S[think_mode]}; effort=${S[reasoning_effort]}; S[think_mode]=default; S[reasoning_effort]=default
        if [[ ${S[configure_thinking]} == true ]]; then
            value=true; [[ $previous != no-think ]] || value=false; yes 'Enable reasoning by default?' "$value"
            if [[ $REPLY == true ]]; then
                S[think_mode]=think; value=$(reasoning_efforts "${S[model]}") || value=''
                if [[ -n $value ]]; then
                    local -a options=('default|Template default'); local found=false
                    for key in $value; do options+=("$key|$key"); [[ $key != "$effort" ]] || found=true; done
                    [[ $found == false ]] || S[reasoning_effort]=$effort
                    choice reasoning_effort 'Reasoning effort' "${options[@]}"
                else printf 'Reasoning-effort support not detected; keeping template default.\n'; fi
            else S[think_mode]=no-think; fi
        fi
        integer max_new 'Max generated tokens (reasoning + answer)' 1
        ask_yes configure_sampling 'Configure default sampling?'
        if [[ ${S[configure_sampling]} == true ]]; then ask temperature 'Temperature (blank = model default)'; integer top_k 'Top K (blank = model default)' 0 999999999 true; ask top_p 'Top P (blank = model default)'; integer seed 'Seed (blank = model default)' 0 999999999 true; fi
    fi
    ask_yes cpu_miss_enabled '5. Experimental CPU expert-miss computation?'
    if [[ ${S[cpu_miss_enabled]} == true ]]; then
        physical_cores; count=$REPLY
        printf 'Physical cores: %s. Requires AVX2 + FMA3 and full-RAM experts.\nLinux uses OS scheduling, not Windows hybrid-core pinning.\n' "$count"
        integer cpu_miss_max 'Maximum CPU misses (1-3)' 1 3
        valid_uint "${S[cpu_miss_cores]}" 1 "$count" || S[cpu_miss_cores]=$((count>2 ? count-2 : 1))
        integer cpu_miss_cores 'CPU compute cores' 1 "$count"
    fi
    if [[ ${S[launch_mode]} == serve ]]; then
        if [[ ${S[mtp_enabled]} == true ]]; then
            [[ ${S[server_parallel]} == 1 || ${S[server_parallel]} == 2 ]] || S[server_parallel]=1
            choice server_parallel '6. Concurrent slots' '1|Single-stream MTP' '2|Two slots, opportunistic MTP'
            if [[ ${S[server_parallel]} == 2 ]]; then S[mtp_verify_tokens]=4; printf 'One active decode uses MTP; two active decodes use ordinary batched decode.\n'; fi
        else integer server_parallel '6. Concurrent slots' 1; fi
    fi
    ask context '7. Context (blank = auto)'
    if [[ ${S[launch_mode]} == serve ]]; then
        if serial_mtp; then S[server_session_cache]=false; printf 'Single-stream text-only MTP disables SSD session caching.\n'
        else ask_yes server_session_cache '8. Cache idle KV sessions on SSD?'; fi
        if [[ ${S[server_session_cache]} == true ]]; then
            yes 'Customize SSD KV settings?' false
            if [[ $REPLY == true ]]; then ask session_cache_dir 'Cache directory' true; integer session_idle_secs 'Spill after idle seconds'; ask session_cache_max 'SSD cache limit' true; integer session_cache_ttl_hours 'Cache TTL hours'; fi
        fi
        while true; do ask server_addr '9. Listen address' true; valid_address "${S[server_addr]}" && break; printf 'Enter an IP:port, such as 127.0.0.1:8080 or [::1]:8080.\n'; done
        ask_yes server_auth 'Enable Bearer API-key authentication?'
    fi
    if [[ ${S[launch_mode]} == bench ]]; then
        choice bench_kind 'Benchmark type' 'decode|Decode' 'prefill|Prefill' 'mixed|Combined turn' 'custom|Custom -p / -n'
        value=1; [[ ${S[bench_kind]} != custom ]] || value=0
        if [[ ${S[bench_kind]} == decode ]]; then S[prompt_tokens]=0; else integer prompt_tokens 'Prompt tokens' "$value"; fi
        if [[ ${S[bench_kind]} == prefill ]]; then S[gen_tokens]=0; else integer gen_tokens 'Generated tokens' "$value"; fi
        choice depth_mode 'Context depth' 'none|None' 'real|Real warmup' 'synthetic|Synthetic'
        if [[ ${S[depth_mode]} == none ]]; then S[depth_tokens]=0; else integer depth_tokens 'Depth tokens' 1; fi
        integer reps 'Repetitions' 1; ask_yes json_output 'Emit JSON?'
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
    add_arg --ctx context
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
        if [[ ${S[bench_kind]} == mixed ]]; then COMMAND+=(--pg "${S[prompt_tokens]},${S[gen_tokens]}"); else COMMAND+=(-p "${S[prompt_tokens]}" -n "${S[gen_tokens]}"); fi
        case ${S[depth_mode]} in real) COMMAND+=(-d "${S[depth_tokens]}") ;; synthetic) COMMAND+=(--synthetic-depth "${S[depth_tokens]}") ;; esac
        COMMAND+=(-r "${S[reps]}"); [[ ${S[json_output]} != true ]] || COMMAND+=(--json)
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
    success=true
)
fetch_url() {
    local url=$1 output=$2 maximum=$3 seconds=${4-600}
    [[ $url == https://api.github.com/repos/Headmaster218/MoE4All/releases\?per_page=100 || $url =~ ^https://github\.com/Headmaster218/MoE4All/releases/download/release-[0-9]+\.[0-9]+\.[0-9]+/MoE4All-Linux-x86_64-v[0-9]+\.[0-9]+\.[0-9]+\.tar\.gz(\.sha256)?$ ]] || return 1
    curl -fLsS --proto '=https' --proto-redir '=https' --connect-timeout 5 --max-time "$seconds" --max-filesize "$maximum" -H 'Accept: application/vnd.github+json' -H 'User-Agent: MoE4All-Linux-Wizard' -o "$output" "$url" || return 1
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
    local root=$1 version=$RELEASE_VERSION temp name="MoE4All-Linux-x86_64-v$RELEASE_VERSION.tar.gz" hash filename extra actual
    [[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || { fail 'Updating supports Linux x86_64 only.'; exit 1; }
    need flock || exit 1; need tar || exit 1; need sha256sum || exit 1
    [[ ! -L $root/.moe4all-update.lock ]] || { fail 'Refusing a symlinked update lock.'; exit 1; }
    exec 9>"$root/.moe4all-update.lock"; flock -n 9 || { fail 'Another update is in progress.'; exit 1; }
    temp=$(mktemp -d "$root/.update-stage.XXXXXX") || exit 1
    trap 'rm -rf -- "$temp"' EXIT
    fetch_url "$RELEASE_CHECKSUM" "$temp/checksum" 4096 || exit 1
    read -r hash filename extra < "$temp/checksum" || exit 1
    [[ $hash =~ ^[0-9a-fA-F]{64}$ && ${filename#\*} == "$name" && -z $extra ]] || { fail 'Malformed release checksum.'; exit 1; }
    fetch_url "$RELEASE_ARCHIVE" "$temp/$name" 1073741824 || exit 1
    actual=$(sha256sum -- "$temp/$name"); actual=${actual%% *}
    [[ $actual == "${hash,,}" ]] || { fail 'Release checksum mismatch; no files changed.'; exit 1; }
    extract_package "$temp/$name" "$temp/files" "$version" || { fail 'Invalid release archive; no files changed.'; exit 1; }
    apply_update "$root" "$temp/files" "$version" || exit 1
    printf 'Update complete. Run the launcher again.\n'
)
check_updates() (
    local mode=$1 current temp
    if ! need curl; then printf 'Update check unavailable; continuing offline.\n'; return; fi
    current=$("$BINARY" --version) || return 0; current=${current##* }
    version_valid "$current" || return 0
    temp=$(mktemp -d); trap 'rm -rf -- "$temp"' EXIT
    if ! fetch_url 'https://api.github.com/repos/Headmaster218/MoE4All/releases?per_page=100' "$temp/releases.json" 8388608 10 || ! json_load "$temp/releases.json" || ! select_release; then
        printf 'Update check unavailable; continuing offline.\n'; return 0
    fi
    if ! version_newer "$RELEASE_VERSION" "$current"; then printf 'Update check: v%s is current\n' "$current"; return 0; fi
    printf 'New engine release: v%s\nhttps://github.com/Headmaster218/MoE4All/releases/tag/release-%s\n' "$RELEASE_VERSION" "$RELEASE_VERSION"
    if [[ -z $RELEASE_ARCHIVE || -z $RELEASE_CHECKSUM || ! -f $ROOT/install-manifest.json || $BINARY != "$ROOT/infr" || -e $ROOT/.git ]]; then
        printf 'Source install or no compatible Linux package: check only; no files changed.\n'
    elif [[ $mode != --update ]]; then printf 'To update explicitly: bash ./Start-INFR-Wizard-Linux.sh --update\n'
    elif [[ $INTERACTIVE != true || $HEADLESS == true ]]; then printf 'Updating requires interactive confirmation; no files changed.\n'
    else yes 'Update this Linux package now?' false; [[ $REPLY != true ]] || download_update "$ROOT"; fi
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
    local key flag value inherited=${INFR_API_KEY-} api_key='' reused=false
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
    CONFIG_DIR=${XDG_CONFIG_HOME:-$HOME/.config}/infr; STATE=$CONFIG_DIR/wizard-state.json
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
    for key in mtp_model vision_projector embedding_model; do
        case $key in mtp_model) value=mtp_enabled ;; vision_projector) value=server_vision ;; embedding_model) value=server_embedding ;; esac
        if [[ -v 'OVERRIDES[$key]' ]]; then S[$value]=false; [[ -z ${S[$key]} ]] || S[$value]=true; fi
    done
    for key in "${!disables[@]}"; do S[$key]=''; S[${disables[$key]}]=false; done
    [[ ! -v 'OVERRIDES[ram_budget]' && ! -v 'OVERRIDES[vram_budget]' ]] || S[configure_memory]=true
    [[ ! -v 'OVERRIDES[kv_type_k]' && ! -v 'OVERRIDES[kv_type_v]' ]] || S[kv_preset]=custom
    if [[ -v 'OVERRIDES[cpu_miss_cores]' ]]; then S[cpu_miss_enabled]=false; [[ ${S[cpu_miss_cores]} == '' || ${S[cpu_miss_cores]} == 0 ]] || S[cpu_miss_enabled]=true; fi
    [[ ! -v 'OVERRIDES[depth_tokens]' ]] || S[depth_mode]=real
    [[ ! -v 'OVERRIDES[prompt_tokens]' && ! -v 'OVERRIDES[gen_tokens]' ]] || S[bench_kind]=custom
    if [[ $NO_API_KEY == true ]]; then S[server_auth]=false; elif [[ -n $inherited ]]; then S[server_auth]=true; fi
    BINARY=''
    for value in "$ROOT/infr" "$ROOT/target/release/infr" "$(command -v infr || true)"; do
        if [[ -f $value && -x $value ]]; then BINARY=$(realpath -- "$value"); break; fi
    done
    [[ -n $BINARY ]] || { fail 'infr not found beside the launcher or under target/release; build infr-cli first.'; return 1; }
    [[ ! -t 0 || $DRY_RUN == true ]] || INTERACTIVE=true
    if [[ -n $UPDATE_MODE ]]; then check_updates "$UPDATE_MODE"; return; fi
    if [[ $INTERACTIVE == true && $SKIP_UPDATE == false && ! ${MOE4ALL_NO_UPDATE_CHECK-} =~ ^(1|true|yes|on)$ ]]; then check_updates --check-update; fi
    if [[ $INTERACTIVE == true && $HEADLESS == false ]]; then
        printf 'MoE4All Linux Launch Wizard\nEnter = previous/default; - = clear\n'
        if [[ -n ${S[model]} && ${#OVERRIDES[@]} == 0 && ${#disables[@]} == 0 ]]; then
            absolute "${S[model]}"; if [[ -f $REPLY ]]; then yes 'Start with previous settings?' true; reused=$REPLY; fi
        fi
        [[ $reused == true ]] || configure
        [[ $NO_API_KEY == false ]] || S[server_auth]=false
    fi
    if [[ ${S[launch_mode]} == serve && ${S[server_auth]} == true ]]; then
        api_key=$inherited
        if [[ $INTERACTIVE == true && $HEADLESS == false ]]; then
            printf 'API key (hidden, Enter reuses inherited key): '; IFS= read -rs value || return 1; printf '\n'; [[ -z $value ]] || api_key=$value
        fi
        [[ -n $api_key ]] || { fail 'API-key authentication requires INFR_API_KEY or hidden input.'; return 1; }
    fi
    for key in model mtp_model vision_projector embedding_model config_path session_cache_dir; do
        [[ -n ${S[$key]} ]] || continue
        if [[ $key == model ]] && hub_reference "${S[$key]}"; then continue; fi
        absolute "${S[$key]}"; S[$key]=$REPLY
    done
    build_command
    printf '\nCommand:\n  '; printf '%q ' "${COMMAND[@]}"; printf '\n'
    if [[ ${S[launch_mode]} == serve && -z $api_key ]] && ! loopback "${S[server_addr]}"; then
        printf 'Warning: network listen address without API-key authentication.\n' >&2
        if [[ $DRY_RUN == false && $HEADLESS == false ]]; then
            [[ $INTERACTIVE == true ]] || { fail 'Unauthenticated network serving requires explicit --yes.'; return 1; }
            yes 'Continue without authentication?' false; [[ $REPLY == true ]] || return 0
        fi
    fi
    [[ $DRY_RUN == false ]] || return 0
    if [[ $HEADLESS == false && $reused == false ]]; then
        [[ $INTERACTIVE == true ]] || { fail 'Launch confirmation requires input or --yes.'; return 1; }
        yes '10. Start now?' true; [[ $REPLY == true ]] || return 0
    fi
    save_state; printf 'Settings saved: %s\n' "$STATE"
    if [[ -n $api_key ]]; then export INFR_API_KEY=$api_key; else unset INFR_API_KEY; fi
    cd -- "$ROOT"; exec "${COMMAND[@]}"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then main "$@"; fi
