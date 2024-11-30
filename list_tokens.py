# Copyright (c) 2024 Rob van Eijk
#
# Permission to use, copy, modify, and/or distribute this software for any 
# purpose with or without fee is hereby granted, provided that the above 
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH 
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY 
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, 
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM 
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR 
# OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR 
# PERFORMANCE OF THIS SOFTWARE.

#!/usr/bin/env python3
import mmap
import struct
import argparse
import os
from typing import Tuple, List, Optional

def hexdump(data: bytes, max_length: int = 40) -> str:
    """Create a compact hex representation of bytes with length limit."""
    hex_str = ' '.join(f'{b:02x}' for b in data)
    if len(hex_str) > max_length:
        return hex_str[:max_length] + '...'
    return hex_str.ljust(max_length)

def find_vocab_section(mm: mmap.mmap) -> Tuple[int, int]:
    """Find the vocabulary section using special tokens."""
    # Look for the first real word after special tokens
    vocab_start = 0
    vocab_end = len(mm)
    
    # First find special tokens to locate the vocab section
    special_tokens = [b'<pad>', b'<eos>', b'<unk>', b'<end_of_turn>']
    special_positions = []
    
    for token in special_tokens:
        pos = mm.find(token)
        if pos != -1:
            special_positions.append(pos)
            print(f"Found special token {token!r} at offset 0x{pos:x}")
    
    if special_positions:
        vocab_start = min(special_positions)
    
    # Find end by looking for long sequence of nulls
    pos = vocab_start
    null_count = 0
    while pos < len(mm):
        if mm[pos:pos+1] == b'\0':
            null_count += 1
            if null_count > 100:
                break
        else:
            null_count = 0
        pos += 1
    
    vocab_end = min(len(mm), pos)
    section_size = vocab_end - vocab_start
    print(f"Vocabulary section: 0x{vocab_start:x} to 0x{vocab_end:x} ({section_size:,} bytes)")
    return vocab_start, vocab_end

def find_next_token(mm: mmap.mmap, start: int, end: int) -> Tuple[Optional[int], Optional[bytes]]:
    """Find next meaningful token and return its position and content."""
    boundary = b'\xe2\x96\x81'  # ▁
    pos = start
    
    while pos < end - 3:
        pos = mm.find(boundary, pos, end)
        if pos == -1:
            return None, None
            
        # Look ahead for actual content
        next_pos = pos + 3  # Skip boundary marker
        content_end = next_pos
        found_content = False
        
        # Look for end of token (next boundary or null)
        while content_end < min(end, next_pos + 100):
            if (mm[content_end:content_end+3] == boundary or 
                mm[content_end:content_end+1] == b'\0'):
                break
            if 32 <= mm[content_end] <= 126:  # Found printable ASCII
                found_content = True
            content_end += 1
        
        token_content = mm[pos:content_end]
        if found_content and len(token_content) > 3:  # More than just boundary marker
            return pos, token_content
            
        pos += 1
    
    return None, None

def analyze_token(mm: mmap.mmap, pos: int, vocab_start: int) -> List[Tuple[int, str]]:
    """Look for potential token IDs before position."""
    ids = []
    for i in range(max(vocab_start, pos - 8), pos):
        if i + 4 <= pos:
            try:
                val_le = struct.unpack('<I', mm[i:i+4])[0]
                val_be = struct.unpack('>I', mm[i:i+4])[0]
                if 0 < val_le < 100000:
                    ids.append((val_le, 'LE'))
                if 0 < val_be < 100000 and val_be != val_le:
                    ids.append((val_be, 'BE'))
            except:
                continue
    return ids

def list_tokens(file_path: str, num_tokens: int, skip: int = 0, window: int = 0):
    """List meaningful tokens with detailed information."""
    file_size = os.path.getsize(file_path)
    print(f"\nAnalyzing file: {file_path}")
    print(f"File size: {file_size:,} bytes")
    
    with open(file_path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        vocab_start, vocab_end = find_vocab_section(mm)
        
        if window > 0:
            print(f"\nShowing window of {window} tokens around position {skip}:")
            skip = max(0, skip - window//2)
            num_tokens = window
        else:
            print(f"\nListing {num_tokens} meaningful tokens (skipping {skip}):")
            
        print("-" * 130)
        print(f"{'#':>4} {'Offset':>10} {'Len':>4} {'Raw Hex':42} {'Token IDs':25} {'Token Text'}")
        print("-" * 130)
        
        pos = vocab_start
        count = 0
        skipped = 0
        
        while count < num_tokens and pos < vocab_end:
            pos, token = find_next_token(mm, pos, vocab_end)
            if pos is None or token is None:
                break
                
            if skipped < skip:
                skipped += 1
                pos += len(token)
                continue
            
            # Get token information
            token_hex = hexdump(token)
            token_ids = analyze_token(mm, pos, vocab_start)
            try:
                token_text = token.decode('utf-8')
            except:
                token_text = '[decode error]'
            
            # Format and print entry
            ids_str = ', '.join(f"{id}({endian})" for id, endian in token_ids)
            print(f"{skip+count:4d} 0x{pos:08x} {len(token):4d} {token_hex} {ids_str:25} {token_text}")
            
            pos += len(token)
            count += 1
        
        print("-" * 130)
        mm.close()

def print_help():
    """Print detailed help information."""
    help_text = """
GGUF Token Lister

This script analyzes and displays tokens from a GGUF model file's vocabulary section.

Usage:
  list_tokens.py [OPTIONS] FILE

Arguments:
  FILE                  Path to the GGUF model file

Options:
  --count N            Number of tokens to display (default: 50)
  --skip N             Number of tokens to skip before starting display (default: 0)
  --window N           Show N tokens centered around the skip position
                      For example, --skip 1000 --window 150 will show 150 tokens
                      centered around position 1000
  -h, --help          Show this help message

Examples:
  # Show first 50 tokens:
  list_tokens.py model.gguf

  # Show 100 tokens starting from position 1000:
  list_tokens.py --count 100 --skip 1000 model.gguf

  # Show 150 tokens around position 1000:
  list_tokens.py --window 150 --skip 1000 model.gguf

Output Format:
  #      - Token number in vocabulary
  Offset - File offset in hexadecimal
  Len    - Length of token in bytes
  Raw Hex- Hexadecimal representation of token data
  IDs    - Potential token IDs found near token
  Text   - Decoded token text
"""
    print(help_text)

def main():
    parser = argparse.ArgumentParser(
        description='List meaningful tokens from GGUF vocabulary',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False)
    
    parser.add_argument('file', nargs='?', help='Path to the GGUF file')
    parser.add_argument('--count', type=int, default=50, 
                      help='Number of tokens to list (default: 50)')
    parser.add_argument('--skip', type=int, default=0,
                      help='Number of tokens to skip (default: 0)')
    parser.add_argument('--window', type=int, default=0,
                      help='Show N tokens centered around skip position')
    parser.add_argument('-h', '--help', action='store_true',
                      help='Show detailed help message')
    
    args = parser.parse_args()
    
    if args.help or not args.file:
        print_help()
        return
    
    list_tokens(args.file, args.count, args.skip, args.window)

if __name__ == '__main__':
    main()
