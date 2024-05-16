import os
import sys
import shutil
import zipfile
import urllib.request
from argparse import Namespace
from app_rvc import (set_logging_level,
                     UVR_MODELS,
                     download_manager,
                     MDX_DOWNLOAD_LINK,
                     mdxnet_models_dir,
                     upload_model_list,
                     ClassVoices,
                     SoniTranslate,
                     logger,
                     TTS_Info)
from cog import BasePredictor, Input, Path as CogPath


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        pass

    def predict(self,
        media_file: CogPath = Input(
            description="Input video",
            default=None),
        link_media: str = Input(
            description="Input link",
            default=None),
        SOURCE_LANGUAGE=Input(
            description="Original language",
            default="Automatic detection"),
        TRANSLATE_AUDIO_TO: str = Input(
            description="Language of translation",
            default=""),
        max_speakers: int = Input(
            description="Maximum number of speakers",
            default=1),
        tts_voice00: str = Input(
            description="Speaker 1",
            default="_XTTS_/AUTOMATIC.wav"),
        tts_voice01: str = Input(
            description="Speaker 2",
            default="_XTTS_/AUTOMATIC.wav"),
        tts_voice02: str = Input(
            description="Speaker 3",
            default="_XTTS_/AUTOMATIC.wav"),
        tts_voice03: str = Input(
            description="Speaker 4",
            default="_XTTS_/AUTOMATIC.wav"),
        tts_voice04: str = Input(
            description="Speaker 5",
            default="_XTTS_/AUTOMATIC.wav"),
        tts_voice05: str = Input(
            description="Speaker 6",
            default="_XTTS_/AUTOMATIC.wav"),
        volume_original_audio: float = Input(
            description="Volume of an original voice track",
            default=0.25),
        volume_translated_audio: float = Input(
            description="Volume of a translated voice track",
            default=1.8),
        output_format_subtitle: str = Input(
            description="Format of a subtitles file",
            default="srt"),
        voiceless_track: bool = Input(
            description="Remove original voice track from the video?",
            default=False
        ),
        voice_imitation: bool = Input(
            description="Try to imitate the original voice?",
            default=False),
        voice_imitation_max_segments: int = Input(
            description="Max number of audio segments for voice imitation",
            default=5
        ),
        voice_imitation_method: str = Input(
            description="Method for voice imitation (freevc or openvoice)",
            default="freevc"
        ),
        soft_subtitles_to_video: bool = Input(
            description="Add soft subtitles to the video?",
            default=False
        ),
        burn_subtitles_to_video: bool = Input(
            description="Add hard subtitles to the video?",
            default=False
        )



    ) -> CogPath:
        set_logging_level("info")

        for id_model in UVR_MODELS:
            download_manager(
                os.path.join(MDX_DOWNLOAD_LINK, id_model), mdxnet_models_dir
            )

        models, index_paths = upload_model_list()
        os.environ["VOICES_MODELS"] = "DISABLE"

        vci = ClassVoices()
        SoniTr = SoniTranslate()

        try:
            from piper import PiperVoice  # noqa

            piper_enabled = True
            logger.info("PIPER TTS enabled")
        except Exception as error:
            logger.warning(str(error))
            piper_enabled = False
            logger.info("PIPER TTS disabled")
        try:
            from TTS.api import TTS  # noqa

            xtts_enabled = True
            logger.info("Coqui XTTS enabled")
            logger.info(
                "In this app, by using Coqui TTS (text-to-speech), you "
                "acknowledge and agree to the license.\n"
                "You confirm that you have read, understood, and agreed "
                "to the Terms and Conditions specified at the following link:\n"
                "https://coqui.ai/cpml.txt."
            )
            os.environ["COQUI_TOS_AGREED"] = "1"
        except Exception as error:
            logger.warning(str(error))
            xtts_enabled = False
            logger.info("Coqui XTTS disabled")

        tts_info = TTS_Info(piper_enabled, xtts_enabled)

        file = SoniTr.multilingual_media_conversion(media_file=media_file,
                                                    link_media=link_media,
                                                    directory_input="",
                                                    YOUR_HF_TOKEN=os.environ["YOUR_HF_TOKEN"],
                                                    SOURCE_LANGUAGE=SOURCE_LANGUAGE,
                                                    TRANSLATE_AUDIO_TO=TRANSLATE_AUDIO_TO,
                                                    max_speakers=max_speakers,
                                                    tts_voice00=tts_voice00,
                                                    tts_voice01=tts_voice01,
                                                    tts_voice02=tts_voice02,
                                                    tts_voice03=tts_voice03,
                                                    tts_voice04=tts_voice04,
                                                    tts_voice05=tts_voice05,
                                                    volume_original_audio=volume_original_audio,
                                                    volume_translated_audio=volume_translated_audio,
                                                    output_format_subtitle=output_format_subtitle,
                                                    voiceless_track=voiceless_track,
                                                    voice_imitation=voice_imitation,
                                                    voice_imitation_method=voice_imitation_method,
                                                    voice_imitation_max_segments=voice_imitation_max_segments,
                                                    soft_subtitles_to_video=soft_subtitles_to_video,
                                                    burn_subtitles_to_video=burn_subtitles_to_video)
        print(f"[+] Translated video generated at {file}")
        return CogPath(file)