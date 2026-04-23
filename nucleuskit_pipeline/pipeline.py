"""
Nucleus-Kit analytics pipeline (offline session processing).

Upstream reference: analyticsEngine/pipelines/HermesAnalysisPipeline.py
"""

from __future__ import annotations

import traceback
from abc import ABC, abstractmethod

from nucleuskit_pipeline.config import resolve_pov_settings
from nucleuskit_pipeline.hermes.processor.cognitionProcessor import computeCognitiveIndexes
from nucleuskit_pipeline.hermes.processor.emotionsProcessor import computeEmotions
from nucleuskit_pipeline.hermes.processor.eventsTools import eventProcessor
from nucleuskit_pipeline.hermes.processor.fileTools import ensure_session_rawdata_layout, prepareDirectory
from nucleuskit_pipeline.hermes.processor.HermesPOVConversion import convertPOVMovieClip
from nucleuskit_pipeline.hermes.processor.metaInfoTools import extractMetaInfo
from nucleuskit_pipeline.hermes.processor.session_video_rotation import ensure_session_video_rotated_180
from nucleuskit_pipeline.events import seedPlaybackAnnotations
from nucleuskit_pipeline.logging_utils import printError, printInfo, printWarning
from nucleuskit_pipeline.position import processGPS, processUWB
from nucleuskit_pipeline.shimmer import computeArousal, computeHeartDynamics


class BasePipeline(ABC):
    """Minimal base from analyticsEngine.pipelines.BasePipeline."""

    def __init__(self, configuration=None):
        self.configured = False
        self.pipeName = "Unknown"

    @abstractmethod
    def processSession(self, jobDetails):
        pass

    def updateSession(self, jobDetails):
        return False

    def configure(self, config_name=None):
        pass

    def get_pipeline_name(self):
        return self.pipeName


class NucleusKitProcessingPipeline(BasePipeline):
    """
    Offline analytics pipeline for processing session data from a local folder.
    """

    def __init__(self, configuration=None):
        super().__init__(configuration)
        self.plot_data = False
        self.processingSteps = []

        if configuration is None:
            self.configureAsDefault()
            self.configured = True
        else:
            printError("[NucleusKitProcessingPipeline] ERROR: Invalid configuration")

    def processSession(self, jobDetails):
        if not self.configured:
            printError("[NucleusKitProcessingPipeline] ERROR: Pipeline not configured")
            return False

        cachepath = jobDetails.path
        ensure_session_rawdata_layout(cachepath)

        pov = resolve_pov_settings(getattr(jobDetails, "pov_config_json", None))
        screen_id = jobDetails.screenID or (pov.screen_id if pov else None)

        printInfo("[NucleusKitProcessingPipeline] Converting movie")
        try:
            if screen_id and pov and pov.data_root and pov.ffmpeg_bin_dir:
                convertPOVMovieClip(screen_id, pov.data_root, pov.ffmpeg_bin_dir)
            elif screen_id:
                printWarning(
                    "[NucleusKitProcessingPipeline] POV screen id set but data_root/ffmpeg not configured; skipping conversion"
                )
        except Exception as e:
            printWarning(f"[NucleusKitProcessingPipeline] WARNING: Converting movie failed: {e}")

        for processor in self.processingSteps:
            try:
                processor(cachepath)
            except BaseException as e:
                printError(
                    f"[NucleusKitProcessingPipeline] ERROR in {processor.__name__}: {type(e).__name__}: {e}"
                )
                printError(f"[NucleusKitProcessingPipeline] Traceback:\n{traceback.format_exc()}")

        return True

    def configureAsDefault(self):
        self.pipeName = "HermesDevPipe"
        self.processingSteps = []
        printInfo(f"[NucleusKitProcessingPipeline] Pipeline: {self.pipeName}")
        printInfo("[NucleusKitProcessingPipeline] Assembling pipeline")

        printInfo("- Adding preparation of directory")
        self.processingSteps.append(prepareDirectory)
        printInfo("- Adding one-time session video 180° rotation (rawData/video.mp4)")
        self.processingSteps.append(ensure_session_video_rotated_180)
        printInfo("- Adding extraction of meta information")
        self.processingSteps.append(extractMetaInfo)

        printInfo("- Adding physiological metrics computation")
        self.processingSteps.append(computeHeartDynamics)
        self.processingSteps.append(computeArousal)

        printInfo("- Adding cognitive metrics computation")
        self.processingSteps.append(computeCognitiveIndexes)

        printInfo("- Adding emotional metrics computation")
        self.processingSteps.append(computeEmotions)

        printInfo("- Adding event processing")
        self.processingSteps.append(eventProcessor)

        printInfo("- Adding playback annotation seeding from events.csv")
        self.processingSteps.append(seedPlaybackAnnotations)

        printInfo("- Adding position processing")
        self.processingSteps.append(processUWB)
        self.processingSteps.append(processGPS)

