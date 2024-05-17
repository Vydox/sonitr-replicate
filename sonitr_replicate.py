from soni_translate.logging_setup import (
    logger,
    set_logging_level,
    configure_logging_libs,
); configure_logging_libs() # noqa
import whisperx
import torch
import os
from soni_translate.audio_segments import create_translated_audio
from soni_translate.text_to_speech import (
    audio_segmentation_to_voice,
    edge_tts_voices_list,
    coqui_xtts_voices_list,
    piper_tts_voices_list,
    create_wav_file_vc,
    accelerate_segments,
)
from soni_translate.translate_segments import translate_text
from soni_translate.preprocessor import (
    audio_video_preprocessor,
    audio_preprocessor,
)
from soni_translate.postprocessor import media_out
from soni_translate.language_configuration import (
    LANGUAGES,
    LANGUAGES_LIST,
    bark_voices_list,
    vits_voices_list,
)
from soni_translate.utils import (
    remove_files,
    upload_model_list,
    download_manager,
    run_command,
    is_audio_file,
    copy_files,
    get_valid_files,
    get_link_list,
    remove_directory_contents,
)
from soni_translate.mdx_net import (
    UVR_MODELS,
    MDX_DOWNLOAD_LINK,
    mdxnet_models_dir,
)
from soni_translate.speech_segmentation import (
    transcribe_speech,
    align_speech,
    diarize_speech,
    diarization_models,
)
from soni_translate.text_multiformat_processor import (
    srt_file_to_segments,
    process_subtitles,
    break_aling_segments,
)
from soni_translate.languages_gui import language_data
import copy
import json
from pydub import AudioSegment
from voice_main import ClassVoices
import hashlib

directories = [
    "downloads",
    "logs",
    "weights",
    "clean_song_output",
    "_XTTS_",
    f"audio2{os.sep}audio",
    "audio",
    "outputs",
]
[
    os.makedirs(directory)
    for directory in directories
    if not os.path.exists(directory)
]


class TTS_Info:
    def __init__(self, piper_enabled, xtts_enabled):
        self.list_edge = edge_tts_voices_list()
        self.list_bark = list(bark_voices_list.keys())
        self.list_vits = list(vits_voices_list.keys())
        self.piper_enabled = piper_enabled
        self.list_vits_onnx = (
            piper_tts_voices_list() if self.piper_enabled else []
        )
        self.xtts_enabled = xtts_enabled

    def tts_list(self):
        self.list_coqui_xtts = (
            coqui_xtts_voices_list() if self.xtts_enabled else []
        )
        list_tts = sorted(
            self.list_edge
            + self.list_bark
            + self.list_vits
            + self.list_vits_onnx
            + self.list_coqui_xtts
        )
        return list_tts


def custom_model_voice_enable(enable_custom_voice):
    os.environ["VOICES_MODELS"] = (
        "ENABLE" if enable_custom_voice else "DISABLE"
    )


def prog_disp(msg):
    logger.info(msg)


def warn_disp(wrn_lang):
    logger.warning(wrn_lang)


class SoniTrCache:
    def __init__(self):
        self.cache = {
            'media': [[]],
            'transcript_align': [],
            'break_align': [],
            'diarize': [],
            'translate': [],
            'subs_and_edit': [],
            'tts': [],
            'acc_and_vc': [],
            'mix_aud': [],
            'output': []
        }

        self.cache_data = {
            'media': [],
            'transcript_align': [],
            'break_align': [],
            'diarize': [],
            'translate': [],
            'subs_and_edit': [],
            'tts': [],
            'acc_and_vc': [],
            'mix_aud': [],
            'output': []
        }

        self.cache_keys = list(self.cache.keys())
        self.first_task = self.cache_keys[0]
        self.last_task = self.cache_keys[-1]

        self.pre_step = None
        self.pre_params = []

    def set_variable(self, variable_name, value):
        setattr(self, variable_name, value)

    def task_in_cache(self, step: str, params: list, previous_step_data: dict):

        self.pre_step_cache = None

        if step == self.first_task:
            self.pre_step = None

        if self.pre_step:
            self.cache[self.pre_step] = self.pre_params

            # Fill data in cache
            self.cache_data[self.pre_step] = copy.deepcopy(previous_step_data)

        self.pre_params = params
        # logger.debug(f"Step: {str(step)}, Cache params: {str(self.cache)}")
        if params == self.cache[step]:
            logger.debug(f"In cache: {str(step)}")

            # Set the var needed for next step
            # Recovery from cache_data the current step
            for key, value in self.cache_data[step].items():
                self.set_variable(key, copy.deepcopy(value))
                logger.debug(
                    f"Chache load: {str(key)}"
                )

            self.pre_step = step
            return True

        else:
            logger.debug(f"Flush next and caching {str(step)}")
            selected_index = self.cache_keys.index(step)

            for idx, key in enumerate(self.cache.keys()):
                if idx >= selected_index:
                    self.cache[key] = []
                    self.cache_data[key] = {}

            # The last is now previous
            self.pre_step = step
            return False

    def clear_cache(self, media, force=False):

        self.cache["media"] = (
            self.cache["media"] if len(self.cache["media"]) else [[]]
        )

        if media != self.cache["media"][0] or force:

            # Clear cache
            self.cache = {key: [] for key in self.cache}
            self.cache["media"] = [[]]

            logger.info("Cache flushed")


def get_hash(filepath):
    with open(filepath, 'rb') as f:
        file_hash = hashlib.blake2b()
        while chunk := f.read(8192):
            file_hash.update(chunk)

    return file_hash.hexdigest()[:18]


class SoniTranslate(SoniTrCache):
    def __init__(self, dev=False):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.result_diarize = None
        self.align_language = None
        self.result_source_lang = None
        self.edit_subs_complete = False
        self.voiceless_id = None
        self.burn_subs_id = None

        logger.info(f"Working in: {self.device}")

    def batch_multilingual_media_conversion(self, *kwargs):
        # logger.debug(str(kwargs))

        media_file_arg = kwargs[0] if kwargs[0] is not None else []

        link_media_arg = kwargs[1]
        link_media_arg = [x.strip() for x in link_media_arg.split(',')]
        link_media_arg = get_link_list(link_media_arg)

        path_arg = kwargs[2]
        path_arg = [x.strip() for x in path_arg.split(',')]
        path_arg = get_valid_files(path_arg)

        edit_text_arg = kwargs[25]
        get_text_arg = kwargs[26]

        is_gui_arg = kwargs[-1]

        kwargs = kwargs[3:]

        media_batch = media_file_arg + link_media_arg + path_arg
        media_batch = list(filter(lambda x: x != "", media_batch))
        media_batch = media_batch if media_batch else [None]
        logger.debug(str(media_batch))

        remove_directory_contents("outputs")

        if edit_text_arg or get_text_arg:
            return self.multilingual_media_conversion(
                media_batch[0], "", "", *kwargs
            )

        if "SET_LIMIT" == os.getenv("DEMO"):
            media_batch = [media_batch[0]]

        result = []
        for media in media_batch:
            # Call the nested function with the parameters
            output_file = self.multilingual_media_conversion(
                media, "", "", *kwargs
            )
            result.append(output_file)

            #if is_gui_arg and len(media_batch) > 1:
            #    gr.Info(f"Done: {os.path.basename(output_file)}")

        return result

    def multilingual_media_conversion(
        self,
        media_file,
        link_media,
        directory_input,
        YOUR_HF_TOKEN,
        preview=False,
        WHISPER_MODEL_SIZE="large-v3",
        batch_size=16,
        compute_type="float16",
        SOURCE_LANGUAGE="Automatic detection",
        TRANSLATE_AUDIO_TO="English (en)",
        min_speakers=1,
        max_speakers=2,
        tts_voice00="en-AU-WilliamNeural-Male",
        tts_voice01="en-CA-ClaraNeural-Female",
        tts_voice02="en-GB-ThomasNeural-Male",
        tts_voice03="en-GB-SoniaNeural-Female",
        tts_voice04="en-NZ-MitchellNeural-Male",
        tts_voice05="en-GB-MaisieNeural-Female",
        video_output_name="",
        AUDIO_MIX_METHOD="Adjusting volumes and mixing audio",
        max_accelerate_audio=2.1,
        acceleration_rate_regulation=False,
        volume_original_audio=0.25,
        volume_translated_audio=1.80,
        output_format_subtitle="srt",
        get_translated_text=False,
        get_video_from_text_json=False,
        text_json="{}",
        diarization_model="pyannote_2.1",
        translate_process="google_translator_batch",
        subtitle_file=None,
        output_type="video (mp4)",
        voiceless_track=False,
        voice_imitation=False,
        voice_imitation_max_segments=3,
        voice_imitation_vocals_dereverb=False,
        voice_imitation_remove_previous=True,
        voice_imitation_method="freevc",
        dereverb_automatic_xtts=True,
        divide_text_segments_by="",
        soft_subtitles_to_video=False,
        burn_subtitles_to_video=False,
        enable_cache=False,
        is_gui=False
        #progress=gr.Progress(),
    ):
        if not YOUR_HF_TOKEN:
            YOUR_HF_TOKEN = os.getenv("YOUR_HF_TOKEN")
            if diarization_model == "disable" or max_speakers == 1:
                if YOUR_HF_TOKEN is None:
                    YOUR_HF_TOKEN = ""
            elif not YOUR_HF_TOKEN:
                raise ValueError("No valid Hugging Face token")
            else:
                os.environ["YOUR_HF_TOKEN"] = YOUR_HF_TOKEN

        if get_translated_text:
            self.edit_subs_complete = False
        if get_video_from_text_json:
            if not self.edit_subs_complete:
                raise ValueError("Generate the transcription first.")

        TRANSLATE_AUDIO_TO = LANGUAGES[TRANSLATE_AUDIO_TO]
        SOURCE_LANGUAGE = LANGUAGES[SOURCE_LANGUAGE]

        if tts_voice00[:2].lower() != TRANSLATE_AUDIO_TO[:2].lower():
            wrn_lang = (
                "Make sure to select a 'TTS Speaker' suitable for"
                " the translation language to avoid errors with the TTS."
            )
            warn_disp(wrn_lang)

        if "_XTTS_" in tts_voice00 and voice_imitation:
            wrn_lang = (
                "When you select XTTS, it is advisable "
                "to disable Voice Imitation."
            )
            warn_disp(wrn_lang)

        if os.getenv("VOICES_MODELS") == "ENABLE" and voice_imitation:
            wrn_lang = (
                "When you use R.V.C. models, it is advisable"
                " to disable Voice Imitation."
            )
            warn_disp(wrn_lang)

        if media_file is None:
            media_file = (
                directory_input
                if os.path.exists(directory_input)
                else link_media
            )
        media_file = (
            media_file if isinstance(media_file, str) else media_file.name
        )

        if not media_file and not subtitle_file:
            raise ValueError(
                "Specifify a media or SRT file in advanced settings"
            )

        if subtitle_file:
            subtitle_file = (
                subtitle_file
                if isinstance(subtitle_file, str)
                else subtitle_file.name
            )

        if not media_file and subtitle_file:
            media_file = "audio_support.wav"
            if not get_video_from_text_json:
                remove_files(media_file)
                srt_data = srt_file_to_segments(subtitle_file)
                total_duration = srt_data["segments"][-1]["end"] + 30.
                support_audio = AudioSegment.silent(
                    duration=int(total_duration * 1000)
                )
                support_audio.export(
                    media_file, format="wav"
                )
                logger.info("Supporting audio for the SRT file, created.")

        if "SET_LIMIT" == os.getenv("DEMO"):
            preview = True
            AUDIO_MIX_METHOD = "Adjusting volumes and mixing audio"
            WHISPER_MODEL_SIZE = "medium"
            logger.info(
                "DEMO; set preview=True; Generation is limited to "
                "10 seconds to prevent CPU errors. No limitations with GPU.\n"
                "DEMO; set Adjusting volumes and mixing audio\n"
                "DEMO; set whisper model to medium"
            )

        # Check GPU
        compute_type = "float32" if self.device == "cpu" else compute_type

        base_video_file = "Video.mp4"
        base_audio_wav = "audio.wav"
        dub_audio_file = "audio_dub_solo.ogg"
        voiceless_audio_file = "audio_Voiceless.wav"
        mix_audio_file = "audio_mix.mp3"
        vid_subs = "video_subs_file.mp4"
        video_output_file = "video_dub.mp4"

        if os.path.exists(media_file):
            media_base_hash = get_hash(media_file)
        else:
            media_base_hash = media_file
        self.clear_cache(media_base_hash, force=(not enable_cache))

        if not get_video_from_text_json:
            self.result_diarize = (
                self.align_language
            ) = self.result_source_lang = None
            if not self.task_in_cache("media", [media_base_hash, preview], {}):
                if is_audio_file(media_file):
                    prog_disp(
                        "Processing audio..."
                    )
                    audio_preprocessor(preview, media_file, base_audio_wav)
                else:
                    prog_disp(
                        "Processing video..."
                    )
                    audio_video_preprocessor(
                        preview, media_file, base_video_file, base_audio_wav
                    )
                logger.debug("Set file complete.")

            if not self.task_in_cache("transcript_align", [
                subtitle_file,
                SOURCE_LANGUAGE,
                WHISPER_MODEL_SIZE,
                compute_type,
                batch_size
            ], {}):
                if subtitle_file:
                    prog_disp(
                        "From SRT file..."
                    )
                    if SOURCE_LANGUAGE == "Automatic detection":
                        raise Exception(
                            "To use an SRT file, you need to specify its "
                            "original language (Source language)"
                        )
                    audio = whisperx.load_audio(base_audio_wav)
                    self.result = srt_file_to_segments(subtitle_file)
                    self.result["language"] = SOURCE_LANGUAGE
                else:
                    prog_disp(
                        "Transcribing..."
                    )
                    SOURCE_LANGUAGE = (
                        None
                        if SOURCE_LANGUAGE == "Automatic detection"
                        else SOURCE_LANGUAGE
                    )
                    audio, self.result = transcribe_speech(
                        base_audio_wav,
                        WHISPER_MODEL_SIZE,
                        compute_type,
                        batch_size,
                        SOURCE_LANGUAGE,
                    )
                logger.debug(
                    "Transcript complete, "
                    f"segments count {len(self.result['segments'])}"
                )

                self.align_language = self.result["language"]
                if not subtitle_file:
                    prog_disp("Aligning...")
                    self.result = align_speech(audio, self.result)
                    logger.debug(
                        "Align complete, "
                        f"segments count {len(self.result['segments'])}"
                    )

            if self.result["segments"] == []:
                raise ValueError("No active speech found in audio")

            if not self.task_in_cache("break_align", [
                divide_text_segments_by,
                self.align_language
            ], {
                "result": self.result,
                "align_language": self.align_language
            }):
                if self.align_language in ["ja", "zh"]:
                    divide_text_segments_by += "|!|?|...|。"
                if divide_text_segments_by:
                    try:
                        self.result = break_aling_segments(
                            self.result,
                            break_characters=divide_text_segments_by,
                        )
                    except Exception as error:
                        logger.error(str(error))

            if not self.task_in_cache("diarize", [
                min_speakers,
                max_speakers,
                YOUR_HF_TOKEN[:len(YOUR_HF_TOKEN)//2],
                diarization_model
            ], {
                "result": self.result
            }):
                prog_disp("Diarizing...")
                diarize_model_select = diarization_models[diarization_model]
                self.result_diarize = diarize_speech(
                    base_audio_wav,
                    self.result,
                    min_speakers,
                    max_speakers,
                    YOUR_HF_TOKEN,
                    diarize_model_select,
                )
                logger.debug("Diarize complete")
            self.result_source_lang = copy.deepcopy(self.result_diarize)

            if not self.task_in_cache("translate", [
                TRANSLATE_AUDIO_TO,
                translate_process
            ], {
                "result_diarize": self.result_diarize
            }):
                prog_disp("Translating...")
                self.result_diarize["segments"] = translate_text(
                    self.result_diarize["segments"],
                    TRANSLATE_AUDIO_TO,
                    translate_process,
                    chunk_size=1800,
                )
                logger.debug("Translation complete")
                logger.debug(self.result_diarize)

        if get_translated_text:

            json_data = []
            for segment in self.result_diarize["segments"]:
                start = segment["start"]
                text = segment["text"]
                speaker = int(segment.get("speaker", "SPEAKER_00")[-1]) + 1
                json_data.append(
                    {"start": start, "text": text, "speaker": speaker}
                )

            # Convert list of dictionaries to a JSON string with indentation
            json_string = json.dumps(json_data, indent=2)
            logger.info("Done")
            self.edit_subs_complete = True
            return json_string.encode().decode("unicode_escape")

        if get_video_from_text_json:

            if self.result_diarize is None:
                raise ValueError("Generate the transcription first.")
            # with open('text_json.json', 'r') as file:
            text_json_loaded = json.loads(text_json)
            for i, segment in enumerate(self.result_diarize["segments"]):
                segment["text"] = text_json_loaded[i]["text"]
                segment["speaker"] = "SPEAKER_0" + str(
                    int(text_json_loaded[i]["speaker"]) - 1
                )

        # Write subtitle
        if not self.task_in_cache("subs_and_edit", [
            copy.deepcopy(self.result_diarize),
            output_format_subtitle,
            TRANSLATE_AUDIO_TO
        ], {
            "result_diarize": self.result_diarize
        }):
            self.sub_file = process_subtitles(
                self.result_source_lang,
                self.align_language,
                self.result_diarize,
                output_format_subtitle,
                TRANSLATE_AUDIO_TO,
            )
            if output_format_subtitle != "srt":
                _ = process_subtitles(
                    self.result_source_lang,
                    self.align_language,
                    self.result_diarize,
                    "srt",
                    TRANSLATE_AUDIO_TO,
                )

        if output_type == "subtitle":
            output = media_out(
                media_file,
                TRANSLATE_AUDIO_TO,
                video_output_name,
                output_format_subtitle,
                file_obj=self.sub_file,
            )
            logger.info(f"Done: {output}")
            return output

        if not self.task_in_cache("tts", [
            TRANSLATE_AUDIO_TO,
            tts_voice00,
            tts_voice01,
            tts_voice02,
            tts_voice03,
            tts_voice04,
            tts_voice05,
            dereverb_automatic_xtts
        ], {
            "sub_file": self.sub_file
        }):
            prog_disp("Text to speech...")
            self.valid_speakers = audio_segmentation_to_voice(
                self.result_diarize,
                TRANSLATE_AUDIO_TO,
                True,
                tts_voice00,
                tts_voice01,
                tts_voice02,
                tts_voice03,
                tts_voice04,
                tts_voice05,
                dereverb_automatic_xtts,
            )

        if not hasattr(vci, 'model_voice_path00'):
            cc_transpose_values = cc_index_values = cc_model_paths = None
        else:
            cc_model_paths = [
                vci.model_voice_path00,
                vci.model_voice_path01,
                vci.model_voice_path02,
                vci.model_voice_path03,
                vci.model_voice_path04,
                vci.model_voice_path05,
                vci.model_voice_path99
            ]

            cc_index_values = [
                vci.file_index200,
                vci.file_index201,
                vci.file_index202,
                vci.file_index203,
                vci.file_index204,
                vci.file_index205,
                vci.file_index299
            ]

            cc_transpose_values = [
                vci.f0method,
                vci.transpose00,
                vci.transpose01,
                vci.transpose02,
                vci.transpose03,
                vci.transpose04,
                vci.transpose05,
                vci.transpose99
            ]

        if not self.task_in_cache("acc_and_vc", [
            max_accelerate_audio,
            acceleration_rate_regulation,
            voice_imitation,
            voice_imitation_max_segments,
            voice_imitation_remove_previous,
            voice_imitation_vocals_dereverb,
            voice_imitation_method,
            os.getenv("VOICES_MODELS"),
            cc_model_paths,
            cc_index_values,
            cc_transpose_values
        ], {
            "valid_speakers": self.valid_speakers
        }):
            audio_files, speakers_list = accelerate_segments(
                    self.result_diarize,
                    max_accelerate_audio,
                    self.valid_speakers,
                    acceleration_rate_regulation,
                )

            # Voice Imitation (Tone color converter)
            if voice_imitation:
                prog_disp(
                    "Voice Imitation..."
                )
                from soni_translate.text_to_speech import toneconverter

                try:
                    toneconverter(
                        copy.deepcopy(self.result_diarize),
                        voice_imitation_max_segments,
                        voice_imitation_remove_previous,
                        voice_imitation_vocals_dereverb,
                        voice_imitation_method,
                    )
                except Exception as error:
                    logger.error(str(error))

            # custom voice
            if os.getenv("VOICES_MODELS") == "ENABLE":
                prog_disp(
                    "Applying customized voices..."
                )
                if cc_model_paths is None:
                    logger.error("Apply the configuration!")

                try:
                    vci(speakers_list, audio_files)
                except Exception as error:
                    logger.error(str(error))

            prog_disp(
                "Creating final translated video..."
            )
            remove_files(dub_audio_file)
            create_translated_audio(
                self.result_diarize, audio_files, dub_audio_file
            )

        # Voiceless track, change with file
        hash_base_audio_wav = get_hash(base_audio_wav)
        if voiceless_track:
            if self.voiceless_id != hash_base_audio_wav:
                from soni_translate.mdx_net import process_uvr_task

                try:
                    # voiceless_audio_file_dir = "clean_song_output/voiceless"
                    remove_files(voiceless_audio_file)
                    uvr_voiceless_audio_wav, _ = process_uvr_task(
                        orig_song_path=base_audio_wav,
                        song_id="voiceless",
                        only_voiceless=True,
                        remove_files_output_dir=False,
                    )
                    copy_files(uvr_voiceless_audio_wav, ".")
                    base_audio_wav = voiceless_audio_file
                    self.voiceless_id = hash_base_audio_wav

                except Exception as error:
                    logger.error(str(error))
            else:
                base_audio_wav = voiceless_audio_file

        if not self.task_in_cache("mix_aud", [
            AUDIO_MIX_METHOD,
            volume_original_audio,
            volume_translated_audio,
            voiceless_track
        ], {}):
            # TYPE MIX AUDIO
            remove_files(mix_audio_file)
            command_volume_mix = f'ffmpeg -y -i {base_audio_wav} -i {dub_audio_file} -filter_complex "[0:0]volume={volume_original_audio}[a];[1:0]volume={volume_translated_audio}[b];[a][b]amix=inputs=2:duration=longest" -c:a libmp3lame {mix_audio_file}'
            command_background_mix = f'ffmpeg -i {base_audio_wav} -i {dub_audio_file} -filter_complex "[1:a]asplit=2[sc][mix];[0:a][sc]sidechaincompress=threshold=0.003:ratio=20[bg]; [bg][mix]amerge[final]" -map [final] {mix_audio_file}'
            if AUDIO_MIX_METHOD == "Adjusting volumes and mixing audio":
                # volume mix
                run_command(command_volume_mix)
            else:
                try:
                    # background mix
                    run_command(command_background_mix)
                except Exception as error_mix:
                    # volume mix except
                    logger.error(str(error_mix))
                    run_command(command_volume_mix)

        if "audio" in output_type or is_audio_file(media_file):
            output = media_out(
                media_file,
                TRANSLATE_AUDIO_TO,
                video_output_name,
                "wav" if "wav" in output_type else (
                    "ogg" if "ogg" in output_type else "mp3"
                ),
                file_obj=mix_audio_file,
            )
            logger.info(f"Done: {output}")
            return output

        hash_base_video_file = get_hash(base_video_file)

        if burn_subtitles_to_video:
            hashvideo_text = [
                hash_base_video_file,
                [seg["text"] for seg in self.result_diarize["segments"]]
            ]
            if self.burn_subs_id != hashvideo_text:
                try:
                    logger.info("Burn subtitles")
                    remove_files(vid_subs)
                    command = f"ffmpeg -i {base_video_file} -y -vf subtitles=sub_tra.srt {vid_subs}"
                    run_command(command)
                    base_video_file = vid_subs
                    self.burn_subs_id = hashvideo_text
                except Exception as error:
                    logger.error(str(error))
            else:
                base_video_file = vid_subs

        if not self.task_in_cache("output", [
            hash_base_video_file,
            hash_base_audio_wav,
            burn_subtitles_to_video
        ], {}):
            # Merge new audio + video
            remove_files(video_output_file)
            run_command(
                f"ffmpeg -i {base_video_file} -i {mix_audio_file} -c:v copy -c:a copy -map 0:v -map 1:a -shortest {video_output_file}"
            )

        output = media_out(
            media_file,
            TRANSLATE_AUDIO_TO,
            video_output_name,
            "mkv" if "mkv" in output_type else "mp4",
            file_obj=video_output_file,
            soft_subtitles=soft_subtitles_to_video,
        )
        logger.info(f"Done: {output}")

        return output


def get_language_config(language_data, language=None, base_key="english"):
    base_lang = language_data.get(base_key)

    if language not in language_data:
        logger.error(
            f"Language {language} not found, defaulting to {base_key}"
        )
        return base_lang

    lg_conf = language_data.get(language, {})
    lg_conf.update((k, v) for k, v in base_lang.items() if k not in lg_conf)

    return lg_conf


if __name__ == "__main__":

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

    lg_conf = get_language_config(language_data, language="english")

    print(SoniTr.multilingual_media_conversion(media_file="how to impress 100 girls in only 5 seconds.mp4",
                                         link_media=None,
                                         directory_input="",
                                         YOUR_HF_TOKEN=os.environ["YOUR_HF_TOKEN"],
                                         TRANSLATE_AUDIO_TO="Russian (ru)",
                                         voice_imitation=False,
                                         #voice_imitation_max_segments=10,
                                         #voice_imitation_method="openvoice",
                                         #tts_voice00="ru-RU-DmitryNeural-Male",
                                         tts_voice00="_XTTS_/AUTOMATIC.wav",
                                         translate_process="google_translator_batch",
                                         burn_subtitles_to_video=False,
                                         voiceless_track=True
                                         ))

