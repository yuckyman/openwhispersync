import logging
from pathlib import Path
from whisperless.audio import AudioProcessor

# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_chapter_processing():
    # paths
    chapter_path = Path("whisperless/files/frankenstein/frankenstein_09_shelley_64kb.mp3")
    
    # process chapter
    processor = AudioProcessor(chapter_path)
    features = processor.process_chapter()
    
    # log feature info
    logger.info(f"Duration: {features.duration:.2f}s")
    logger.info(f"Sample rate: {features.sample_rate}Hz")
    logger.info(f"Number of MFCCs: {features.mfcc.shape}")
    logger.info(f"Number of silent regions: {len(features.silent_regions)}")
    
    # log first few silent regions
    for i, (start, end) in enumerate(features.silent_regions[:5]):
        logger.info(f"Silent region {i}: {start:.2f}s to {end:.2f}s (duration: {end-start:.2f}s)")

if __name__ == "__main__":
    test_chapter_processing() 