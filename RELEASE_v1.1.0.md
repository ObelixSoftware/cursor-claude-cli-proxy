# v1.1.0 Release Notes

## 🎉 Image Support

This release brings **full image support** to the Cursor Claude CLI Proxy. The proxy now accepts and forwards images to Claude in multiple formats, enabling vision capabilities in your Cursor editor.

### What's New

#### Image Input Support
- ✅ **Data URLs** (`data:image/png;base64,...`) - embed images directly in requests
- ✅ **HTTPS URLs** (`https://...`) - reference remote images
- ✅ **Multiple formats** - PNG, JPEG, GIF, WebP supported
- ✅ **Size handling** - images up to 5 MiB accepted
- ✅ **Multiple images per request** - send multiple images in a single message

#### Implementation Details
The proxy now:
- Parses `image_url`, `input_image`, and `image` content parts
- Forwards vision input to Claude using stream-json format when images are present
- Validates image URLs and base64 encoding before sending to Claude
- Handles image data transparently through both Chat Completions and Responses APIs

#### Updated CLI Invocation
When images are present, the proxy invokes Claude with:
```
claude --print --safe-mode \
       --input-format stream-json \
       --verbose \
       --output-format stream-json \
       [... other args ...]
```

### Version Bump
- **From:** 0.1.0 (text-only)
- **To:** 1.1.0 (text + images)

### Breaking Changes
None. This is a backward-compatible enhancement.

### Known Limitations
- **Audio and files still not supported** - `input_audio`, `audio`, `input_file`, `file`, and `file_url` parts are rejected with HTTP 400
- **Streaming is buffered** - images are included in the buffered output; token-by-token streaming is not available
- **Size limits apply** - individual images must be under 5 MiB; total request must be under `CLI_PROXY_MAX_REQUEST_BYTES` (default 4 MiB)

### Testing
Image support has been verified with:
- 1x1 PNG (data URL) - minimal valid test case
- HTTPS image URLs - remote content
- Base64 decoding validation - malformed input rejection
- Stream-json format - Claude CLI integration

Example curl test:
```bash
curl -s -H "Authorization: Bearer $CLI_PROXY_TOKEN" \
     -H 'Content-Type: application/json' \
     "http://127.0.0.1:8787/v1/chat/completions" \
     -d '{
       "model": "claude-cli-sonnet",
       "messages": [{
         "role": "user",
         "content": [
           {"type": "text", "text": "What is in this image?"},
           {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="}}
         ]
       }]
     }' | jq -r '.choices[0].message.content'
```

### Documentation
- See [README.md §3 Limitations](README.md#3-limitations) for full image handling details
- See [README.md §9 Testing with curl](README.md#9-testing-with-curl) - "Image input is accepted" example
- See [Module layout §`src/cli_proxy/images.py`](README.md#module-layout) for implementation details

### Files Changed
- `pyproject.toml` - version bumped from 0.1.0 to 1.1.0
- `src/cli_proxy/images.py` - image parsing and stream-json encoding
- `src/cli_proxy/claude_runner.py` - conditional stream-json format selection
- `src/cli_proxy/normalize.py` - image part processing
- Related test coverage for vision requests

### Performance Impact
- Minimal - image handling only activates when images are present in the request
- No performance regression for text-only requests
- Stream-json format (when images present) has comparable latency to json format

### Upgrade Path
1. Update your installation: `pip install -e ".[dev]"` (or rebuild your venv)
2. No configuration changes required
3. Existing Cursor setups continue to work unchanged
4. Start sending images via the Chat Completions or Responses APIs

---

**Status:** Experimental (unchanged)  
**Compatibility:** Requires Claude Code CLI 2.1.231 or later  
**Python:** 3.12+

For questions or issues, refer to the [Troubleshooting](README.md#13-troubleshooting) section in the README.
