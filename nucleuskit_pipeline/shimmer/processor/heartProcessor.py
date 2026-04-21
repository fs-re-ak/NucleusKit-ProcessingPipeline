"""
Heart rate dynamics from Shimmer PPG.

Author(s):
    Fred Simard (fs@re-ak.com), ©RE-AK Technologies Inc.
    Winter 2026
"""

import os

from nucleuskit_pipeline.logging_utils import printInfo, printWarning
from nucleuskit_pipeline.shimmer.localTools.HeartRateUtils import extractHRVandBPM_v2
from nucleuskit_pipeline.shimmer.localTools.PPGUtils import loadPPG, conditionPPG, writePPGEventsFeatures


def computeHeartDynamics(recPath, show=True):
    """
    Compute heart rate dynamics from PPG data.

    Args:
        recPath: Path to the recording directory
        show: Whether to display plots (default: True)
    """
    printInfo("[physioTools] Computing Heart Dynamics")

    # Skip if already done (caching)
    bpm_hrv_path = os.path.join(recPath, "results", "BPM_HRV.csv")
    if os.path.isfile(bpm_hrv_path):
        printInfo("[physioTools] Heart dynamics already computed, using cached results")
        return

    # load PPG
    ppgtime, ppgRec = loadPPG(recPath + "/rawData/", showLoadedSignalInfo=False)


    if ppgRec is not None:

        if len(ppgRec) > 250:  # a very short recording breaks the pipeline
            # step 2 - Condition signal
            ppgRec_conditioned = conditionPPG(ppgRec)
            # step 3 - Compute HRV and BPM
            tt, HRV, BPM = extractHRVandBPM_v2(ppgRec_conditioned, recPath, experimental=True)


            #for peak in tt:
            #    plt.axvline(x=peak, color='k')

            #plt.plot(ppgtime, ppgRec, color='b')
            #plt.plot(ppgRec_conditioned, color='r')
            #plt.show()
            #plt.savefig(os.path.join(recPath, "features", "ppg", "conditionned_ppg.png"))

            """
            if show:
                plt.show()
            else:
                plt.close()
            """

            if tt is not None:
                writePPGEventsFeatures(tt, BPM, HRV, recPath)
                printInfo("[physioTools] Heart dynamics computation completed")
            else:
                printWarning("[physioTools] No valid HRV/BPM computed")

        else:
            printWarning("[physioTools] Shimmer recording too short to compute PPG")
    else:
        printWarning("[physioTools] PPG data is None, skipped")
