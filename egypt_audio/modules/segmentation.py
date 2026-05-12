import numpy as np
from typing import List, Dict, Any

class IntelligentSegmenter:
    """
    Groups VAD segments into semantic chunks based on duration and pause boundaries.
    """
    
    def __init__(self, min_duration: float = 1.0, max_duration: float = 20.0):
        self.min_duration = min_duration
        self.max_duration = max_duration
        
    def chunk_audio(self, audio: np.ndarray, sr: int, speech_segments: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        Groups small segments into bigger chunks until max_duration is hit.
        """
        chunks = []
        if not speech_segments:
            return chunks
            
        current_chunk_start = speech_segments[0]['start']
        current_chunk_end = speech_segments[0]['end']
        
        for i in range(1, len(speech_segments)):
            seg = speech_segments[i]
            
            # Check if merging this segment would exceed max duration
            if (seg['end'] - current_chunk_start) > self.max_duration:
                # Finalize current chunk
                chunks.append({
                    "start": current_chunk_start,
                    "end": current_chunk_end,
                    "duration": round(current_chunk_end - current_chunk_start, 3)
                })
                # Start new chunk
                current_chunk_start = seg['start']
                current_chunk_end = seg['end']
            else:
                # Merge
                current_chunk_end = seg['end']
                
        # Final chunk
        chunks.append({
            "start": current_chunk_start,
            "end": current_chunk_end,
            "duration": round(current_chunk_end - current_chunk_start, 3)
        })
        
        # Filtering
        chunks = [c for c in chunks if c['duration'] >= self.min_duration]
        
        return chunks

if __name__ == "__main__":
    # Test
    seg = IntelligentSegmenter()
    mock_segs = [{"start": 0.0, "end": 2.0}, {"start": 3.0, "end": 6.0}]
    print(seg.chunk_audio(np.zeros(16000), 16000, mock_segs))
