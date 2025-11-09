#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.1.1),
    on October 30, 2025, at 22:12
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""
# Change this before you run the script
BOARD_CONFIG_COM_PORT = "COM8"
DATA_FOLDER_NAME = "new_data"
NUMBER_OF_REPS = "5"
import uuid

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding


import psychopy.iohub as io
from psychopy.hardware import keyboard

from multiprocessing import Process, Value, Event, Manager

deviceManager = hardware.DeviceManager()

# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.1.1'
expName = 'untitled'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []

expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),

    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = (1024, 768)

class RecordingProcess(Process):
    """
    Process that continuously records data from the board when the shared recording_flag is True.
    """
    def __init__(self, record_save_path, recording_flag=False, shared_word=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.record_save_path = record_save_path  # shared path
        self.recording_flag = Value('b', recording_flag)  # shared flag
        self.shared_word = shared_word  # shared word variable
        self._running = Event()
        self._running.set()  # _running indicates whether to continue looping

    def run(self):
        # Instantiate the board connection in the child process.
        from knight_board_init import KnightBoard
        from brainflow.data_filter import DataFilter, DetrendOperations, FilterTypes
        import numpy as np

        self.kb = KnightBoard(BOARD_CONFIG_COM_PORT, 8)
        self.kb.start_stream()
        self.board_shim = self.kb.board_shim
        self.sampling_rate = self.kb.sr
        self.eeg_channels = self.kb.eeg_channels
        self.index = 0
        

        prev_recording_flag = False  # Helps us detect the rising edge.

        while self._running.is_set():
            current_flag = self.recording_flag.value

            if current_flag:
                if not prev_recording_flag:
                    # Use the shared word for the filename
                    current_word = self.shared_word.value
                    random_uuid = uuid.uuid4()
                    self.record_save_path = f"./{DATA_FOLDER_NAME}/{current_word}_{random_uuid}.csv"
                    
                    data = self.board_shim.get_current_board_data(625)
                    if data.shape[1] > 0:
                        for count, channel in enumerate(self.eeg_channels):
                            # Process each channel
                            DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
                            DataFilter.perform_bandstop(
                                data[channel], self.sampling_rate, 39.5, 40.5, 4,
                                FilterTypes.BUTTERWORTH_ZERO_PHASE, 0
                            )
                            DataFilter.perform_bandpass(
                                data[channel], self.sampling_rate, 3.0, 48.0, 2,
                                FilterTypes.BUTTERWORTH_ZERO_PHASE, 0
                            )
                            DataFilter.perform_bandstop(
                                data[channel], self.sampling_rate, 49.5, 50.5, 4,
                                FilterTypes.BUTTERWORTH_ZERO_PHASE, 0
                            )
                            DataFilter.perform_bandstop(
                                data[channel], self.sampling_rate, 59.0, 61.0, 4,
                                FilterTypes.BUTTERWORTH_ZERO_PHASE, 0
                            )
                        # Convert to list of rounded values
                        channel_data = [
                            np.round(data[channel], decimals=7).tolist()
                            for channel in self.eeg_channels
                        ]
                        # Replace any negative zero values
                        for i in range(len(channel_data)):
                            channel_data[i] = [0 if x == 0 else x for x in channel_data[i]]
                        transposed_data = np.array(channel_data).T

                        with open(self.record_save_path, 'a') as f:
                            np.savetxt(f, transposed_data, delimiter=",", fmt="%.7f")
                        print("dataRECORDED")
            # Update previous flag state
            prev_recording_flag = current_flag

        self.kb.stop_stream()

    def stop(self):
        self._running.clear()


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\abdal\\OneDrive\\Desktop\\old desktop\\EEG DEVICE\\github_neuropawn\\Neural-Speech-Decoding\\DataCollectionExperiment.py',
        savePickle=False, saveWideText=False,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp



def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate

    msg = visual.TextStim(win, text='Setting up KnightBoard, please wait...', pos=(0, 0), color='white')
    msg.draw()
    win.flip()

    # --- Setup recording thread and knight board ---
    global recording_process
    recording_process = RecordingProcess(
        record_save_path="./data/recording.csv",
        shared_word=shared_word
    )
    recording_process.start()
    core.wait(25)

    win.flip()


    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    # create speaker 'auditory_stimulus'
    deviceManager.addDevice(
        deviceName='auditory_stimulus',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1,
        resample=True,
        latencyClass=1,
    )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    test_case_number: Current test case number
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "trial" ---
    auditory_stimulus = sound.Sound(
        'A', 
        secs=0.5, 
        stereo=True, 
        hamming=True, 
        speaker='auditory_stimulus',    name='auditory_stimulus'
    )
    auditory_stimulus.setVolume(1.0)
    word_of_interest = visual.TextStim(win=win, name='word_of_interest',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.3, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0)
    cool_off_text = visual.TextStim(win=win, name='cool_off_text',
        text='COOL OFF',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.2, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0)
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=NUMBER_OF_REPS, 
        method='fullRandom', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('words.csv'), 
        seed=None, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        trials.status = STARTED
        if hasattr(thisTrial, 'status'):
            thisTrial.status = STARTED
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "trial" ---
        # create an object to store info about Routine trial
        trial = data.Routine(
            name='trial',
            components=[auditory_stimulus, word_of_interest, cool_off_text],
        )
        trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        auditory_stimulus.setSound("A", secs=0.5, hamming=True)
        auditory_stimulus.setVolume(2.0, log=False)
        auditory_stimulus.seek(0)
        word_of_interest.setText(words)
        shared_word.value = words  # <-- Add this line to update the shared word
        # store start times for trial
        trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trial.tStart = globalClock.getTime(format='float')
        trial.status = STARTED
        thisExp.addData('trial.started', trial.tStart)
        trial.maxDuration = None
        # keep track of which components have finished
        trialComponents = trial.components
        for thisComponent in trial.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial" ---
        trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 6.0:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *auditory_stimulus* updates
            
            # if auditory_stimulus is starting this frame...
            if auditory_stimulus.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                auditory_stimulus.frameNStart = frameN  # exact frame index
                auditory_stimulus.tStart = t  # local t and not account for scr refresh
                auditory_stimulus.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('auditory_stimulus.started', tThisFlipGlobal)
                # update status
                auditory_stimulus.status = STARTED
                auditory_stimulus.play(when=win)  # sync with win flip
            
            # if auditory_stimulus is stopping this frame...
            if auditory_stimulus.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > auditory_stimulus.tStartRefresh + 0.5-frameTolerance or auditory_stimulus.isFinished:
                    # keep track of stop time/frame for later
                    auditory_stimulus.tStop = t  # not accounting for scr refresh
                    auditory_stimulus.tStopRefresh = tThisFlipGlobal  # on global time
                    auditory_stimulus.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'auditory_stimulus.stopped')
                    # update status
                    auditory_stimulus.status = FINISHED
                    auditory_stimulus.stop()
            
            # *word_of_interest* updates
            
            # if word_of_interest is starting this frame...
            if word_of_interest.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                word_of_interest.frameNStart = frameN  # exact frame index
                word_of_interest.tStart = t  # local t and not account for scr refresh
                word_of_interest.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(word_of_interest, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'word_of_interest.started')
                # update status
                word_of_interest.status = STARTED
                word_of_interest.setAutoDraw(True)
            
            # if word_of_interest is active this frame...
            if word_of_interest.status == STARTED:
                # update params
                recording_process.recording_flag.value = False
                pass
            
            # if word_of_interest is stopping this frame...
            if word_of_interest.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > word_of_interest.tStartRefresh + 4-frameTolerance:
                    # keep track of stop time/frame for later
                    word_of_interest.tStop = t  # not accounting for scr refresh
                    word_of_interest.tStopRefresh = tThisFlipGlobal  # on global time
                    word_of_interest.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'word_of_interest.stopped')
                    # update status
                    word_of_interest.status = FINISHED
                    word_of_interest.setAutoDraw(False)
                    recording_process.recording_flag.value = True
            
            # *cool_off_text* updates
            
            # if cool_off_text is starting this frame...
            if cool_off_text.status == NOT_STARTED and tThisFlip >= 4.5-frameTolerance:
                # keep track of start time/frame for later
                cool_off_text.frameNStart = frameN  # exact frame index
                cool_off_text.tStart = t  # local t and not account for scr refresh
                cool_off_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cool_off_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cool_off_text.started')
                # update status
                cool_off_text.status = STARTED
                cool_off_text.setAutoDraw(True)
            
            # if cool_off_text is active this frame...
            if cool_off_text.status == STARTED:
                # update params
                pass
            
            # if cool_off_text is stopping this frame...
            if cool_off_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cool_off_text.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    cool_off_text.tStop = t  # not accounting for scr refresh
                    cool_off_text.tStopRefresh = tThisFlipGlobal  # on global time
                    cool_off_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cool_off_text.stopped')
                    # update status
                    cool_off_text.status = FINISHED
                    cool_off_text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=trial,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                trial.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial" ---
        for thisComponent in trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trial
        trial.tStop = globalClock.getTime(format='float')
        trial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trial.stopped', trial.tStop)
        auditory_stimulus.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if trial.maxDurationReached:
            routineTimer.addTime(-trial.maxDuration)
        elif trial.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-6.000000)
        # mark thisTrial as finished
        if hasattr(thisTrial, 'status'):
            thisTrial.status = FINISHED
        # if awaiting a pause, pause now
        if trials.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trials.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials'
    trials.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    from multiprocessing import freeze_support, Manager
    freeze_support()  # Optional, but recommended for frozen executables

    manager = Manager()
    shared_word = manager.Value('u', '')  # Unicode string for the word

    # call all functions in order

    thisExp = setupData(expInfo=expInfo)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float',
    )
    quit(thisExp=thisExp, win=win)
