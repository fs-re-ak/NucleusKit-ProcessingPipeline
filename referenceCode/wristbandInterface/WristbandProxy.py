
#!/usr/bin/env python3
"""
Proxy 
    - 

Roles:
    - Muse is wrapped up on top of a file socket, shared with a supporting daemon program
    
Design:
    - data reception happens within a listening thread, unpack and transfer to adapter 

TODO:
    - send infor to daemon, through socket not implemented 
    
Author(s):
    Fred Simard (fs@re-ak.com), ©RE-AK Technologies Inc.
    Spring 2018
"""
from abc import ABC, abstractmethod
import numpy as np
import os
import math
import socket
import sys
from configs.PeripheralConfig import PeripheralConfig
from configs.HardwareConfigs import HWConfigs
from threading import Thread
from time import sleep, time
import subprocess
from threading import Event
import shlex
import signal
import serial
import struct
import datetime
 
"""
ProxyFactory, arbiter the creation of the proxy, based on requested configuration.

Public:
    - factory(), static method to get a reference to requested object
    - thishelp(), static method that shows (console) list of available configurations
"""
class WristbandProxyFactory(object):

    """
        Name: factory
        Descr: return a reference to requested object
        Parameters:
            configuration: string of value: {"PCONFIG1"}
            manager: reference to the manager
            
        Return:
            reference to requested proxy, None otherwise
    """
    def factory(configuration,manager):
        
        if configuration == "SHIMMER-IO": return ProxyShimmerIO(manager)
        if configuration == "SHIMMER-SERIAL": return ProxyShimmerSerial(manager)
        else: 
            print('[WristbandProxy.py] invalid factory configuration')
            return None
        
    factory = staticmethod(factory)

    """
        Name: help
        Descr: shows (console) list of available configurations
    """
    def thishelp():
        print("Available configurations:")
        print("     - MUSE")
        print("     - FAKE-MUSE")
        
    thishelp = staticmethod(thishelp) 
 
 
"""
WristbandProxyBase, abstract class that defines the base of all Proxies

Public:
    - __init__(), base constructor (require reference to manager)
    - activate(), to flip an object to active state
    - deactivate(), to flip an object to non-active state
    - attachAdapter(), to attach adapter to proxy
    - doSomething(), abstract method exemple(template only)
"""
class WristbandProxyBase(ABC):
    
    adapter = None
    manager = None
 
    """
        Name: __init__
        Descr: base constructor
        Parameters:
            managerHandle: reference to manager
    """
    def __init__(self, managerHandle):
        self.manager = managerHandle
        super().__init__()

    """
        Name: attachAdapter
        Descr: copy reference to adapter, should be called before object is made active
        Parameters:
            adapterHandle: reference to adapter
            
    """
    def attachAdapter(self, adapterHandle):
        self.adapter = adapterHandle
        pass

    @abstractmethod
    def onEvent(self, event, data):
        pass
    
    @abstractmethod    
    def connect(self):
        pass
        
    @abstractmethod    
    def send(self):
        pass
        
    @abstractmethod    
    def disconnect(self):
        pass


"""
ProxyMuse, proxy for muse device, connected through unix socket

Public:
    - doSomething(), method overriding exemple (template only)
"""
class ProxyShimmerIO(WristbandProxyBase):

    # default location of socket file
    serverAddress = '/tmp/wristband_socket'
    connection = None
    recording = False
    sock = None
    lstData = np.zeros((1048576, 6))
    idxData = 0
    rfCommProc = None
    monoProc = None
    rfCommMonitor = None
    rfCommPollInterval = 5
    rfCommId = 5
    evRfComm = None
    
    """
        Name: __init__
        Descr: constructor
        Parameters:
            managerHandle: reference to manager
    """
    def __init__(self, managerHandle):
        super().__init__(managerHandle)
        self.evRfComm = Event()
        
        # Make sure the socket does not already exist
        try:
            os.unlink(self.serverAddress)
        except OSError:
            if os.path.exists(self.serverAddress):
                raise

    """
        Name: doSomething
        Descr: shows a message in console
    """
    def connect(self):
        
        if self.sock is None:
            self.evRfComm.clear()
            print('[WristbandProxy.py] Shimmer: connect')
            print('[WristbandProxy.py] Shimmer: starting Shimmer process')
            
            # remove any existing socket file
            try:
                os.remove(self.serverAddress)
            except:
                pass

            os.system("sudo hciconfig hci0 reset && sleep 1")
            os.system("sudo rfcomm -i hci0 release {} && sleep 2".format(self.rfCommId))

            hwAddr = PeripheralConfig().getConfig(HWConfigs.PERIPHERAL_WRIST_SHIMMER)

            if hwAddr is None:
                print('[WristbandProxy.py] Wristband: Invalid hw_addr \'{}\''.format(hwAddr))
                sys.exit()

            print('[WristbandProxy.py] Wristband: trying to connect with \'{}\''.format(hwAddr))
            rfCommCmd = "rfcomm -i hci0 connect {} {} 1".format(self.rfCommId, hwAddr)

            os.system(rfCommCmd)

            #while True:
            #    #subprocess.Popen(shlex.split("hciconfig hci0 reset")).communicate()
            #    self.rfCommProc = subprocess.Popen(shlex.split(rfCommCmd))
            #    bootstrap =  Thread(target=self.rfcommBootstrap, args=(5,))
            #    bootstrap.start()
            #    bootstrap.join()
            #    if not self.evRfComm.wait(1):
            #        self.rfCommProc.send_signal(signal.SIGINT)
            #        self.rfCommProc = None
            #    else:
            #        break

            # rfcomm socket created successfully
            print('[WristbandProxy.py] rfcomm socket created successfully')

            #self.monoProc = subprocess.Popen(shlex.split("mono /home/rock64/locApplication/supportDaemons/ShimmerConsoleAppExample.exe"), stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell=True)
            #self.monoProc = subprocess.Popen(["mono /home/rock64/locApplication/supportDaemons/ShimmerConsoleAppExample.exe"], stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell=True)
            os.system("mono /home/vizia/locApplication/supportDaemons/ShimmerConsoleAppExample.exe &")

            print('[WristbandProxy.py] shimmer process started')
            self.rfCommMonitor = Thread(target=self.rfCommWatchdog,)
            self.rfCommMonitor.daemon = True
            self.rfCommMonitor.start()

            # wait for socket to be opened
            while not os.path.exists(self.serverAddress):
                print('[WristbandProxy.py] waiting for socket')
                sleep(1)
                
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                print('connecting to %s' % self.serverAddress)
                self.sock.connect(self.serverAddress)
            except:
                print('[WristbandProxy.py] Shimmer: connect to socket failed')
                sys.exit(0)

        # start thread to service client socket
        self.t = Thread(target=self.socketServiceTask,) 
        self.t.daemon = True
        self.t.start()
        return True

    def rfcommBootstrap(self, pollTO):
        if self.rfCommProc is None:
            print('[WristbandProxy.py] Rfcomm Proc was None!')
        rfCommDevId = "/dev/rfcomm{}".format(self.rfCommId)
        curPollTO = 1
        while curPollTO <= pollTO:
            if os.path.exists(rfCommDevId):
                self.evRfComm.set()
                return
            sleep(curPollTO)
            curPollTO = curPollTO + 1
        pass

    def _isMonoIsAlive(self):
        nbInstances = len(os.popen('pidof mono').read().strip().split())
        if nbInstances > 0:
            return True
        return False

    def _killallMono(self):
        listInstances = os.popen('pidof mono').read().strip().split()
        if len(listInstances) > 0:
            for pid in listInstances:
                os.system("sudo kill -9 " + pid)
        sleep(1)
        self._isMonoIsAlive()

    def rfCommWatchdog(self):
        print('[WristbandProxy.py] shimmer watchdog started')
        rfCommDevId = "/dev/rfcomm{}".format(self.rfCommId)

        while os.path.exists(rfCommDevId) and self._isMonoIsAlive():
            sleep(self.rfCommPollInterval)
        print('[WristbandProxy.py] Shimmer: device disconnected')

        self._killallMono()

        if self.sock is not None:
            self.sock.close()
            self.sock = None
        self.manager.wristbandDisconnected()
    
    """
        Name: socketServiceTask
        Descr: task that services the socket
    """
    def socketServiceTask(self):
        # Receive the data in small chunks and retransmit it
        data = None
        while True:
            try:
                data = self.sock.recv(1024)
            except AttributeError:  # socket was closed and set to None
                print('[WristbandProxy.py] Shimmer: socket closed, receiver thread closing')
                return
            if not self.recording:
                continue
            
            if data:
                data = data.decode("utf-8")
                array = data.split('\n')[:-1]
                
                for y in array:
                    myData = [float(x) for x in y.split()]

                    if len(myData) == 6:
                        self.lstData[self.idxData,:] = myData
                        self.idxData = self.idxData + 1
                        #self.adapter.dataRecved(np.array(myData))
            else:
                print('[WristbandProxy.py] Shimmer: socket activity, no data received')
                self.disconnect()
                break

    def onEvent(self, event, **data):
        if event == "RECORDING":
            if data is None:
                raise ValueError("malformed event detected, no data associated")
            if "state" not in data:
                raise ValueError("malformed event detected, no state is given")

            if not self.recording:
                self.idxData = 0
                self.lstData = np.zeros((1048576, 6))

            self.recording = data["state"]

        elif event == "POST-PROCESS":
            if self.recording:
                raise ValueError("Post process is called while recording")
            return "shimmer-3", self.idxData, self.lstData
        elif event == "STATUS":
            if self.rfCommProc is None or not self._isMonoIsAlive():
                return "shimmer-3", ["DISCONNECTED"]
            elif self.recording:
                return "shimmer-3", ["CONNECTED", "RECORDING"]
            else:
                return "shimmer-3", ["CONNECTED"]
        pass
        
    """
        Name: send
        Descr: not suported, yet
    """
    def send(self):
        print("Send not supported, yet")
        pass
        
    """
        Name: disconnect
        Descr: disconnect, by killing daemon process
    """
    def disconnect(self):
        self._killallMono()
        pass


class ProxyShimmerSerial(WristbandProxyBase):

    evRfComm = None
    recording = False

    def __init__(self, managerHandle):
        super().__init__(managerHandle)
        self.evRfComm = Event()

    """
        Blocking call, will return only when shimmer is connected
    """
    def _connectShimmer(self, hci='hci0', rfCommId=5):

        port = f"/dev/rfcomm5{rfCommId}"

        # get shimmer bt-mac address
        
        try:
            hwAddr = PeripheralConfig().getConfig(HWConfigs.PERIPHERAL_WRIST_SHIMMER)
            if hwAddr is None:
                print('[WristbandProxy.py] Wristband: Invalid hw_addr \'{}\''.format(hwAddr))
                sys.exit()
        except:
            print("using default Shimmer address - this shouldn't happen")
            hwAddr = "00:06:66:E2:89:AF"

        # reset hci
        os.system(f"sudo hciconfig {hci} reset && sleep 1")
        os.system(f"sudo rfcomm -i {hci} release {rfCommId} >/dev/null 2>&1 && sleep 2")

        rfCommCmd = f"rfcomm -i hci0 connect {rfCommId} {hwAddr} 1 &"

        while True:
            subprocess.Popen(shlex.split(f"hciconfig {hci} reset")).communicate()
            self.rfCommProc = subprocess.Popen(shlex.split(rfCommCmd))
            bootstrap = Thread(target=self._rfcommBootstrap, args=((5, 5),))
            bootstrap.start()
            bootstrap.join()
            if not self.evRfComm.wait(1):
                self.rfCommProc.send_signal(signal.SIGINT)
                self.rfCommProc = None
            else:
                break
    
    def _rfcommBootstrap(self, params):
        pollTO, rfCommId = params
        if self.rfCommProc is None:
            print('[WristbandProxy.py] Rfcomm Proc was None!')
        rfCommDevId = f"/dev/rfcomm{rfCommId}"
        curPollTO = 1
        while curPollTO <= pollTO:
            if os.path.exists(rfCommDevId):
                self.evRfComm.set()
                return
            sleep(curPollTO)
            curPollTO = curPollTO + 1
        pass

    def connect(self):
        # connect - blocking call, shimmer should have been paired in the past
        self._connectShimmer()

        self._serial = serial.Serial("/dev/rfcomm5", 115200)
        self._serial.flushInput()
        print("[WristbandProxy.py] Shimmer: port opening, done.")
        # send the set sensors command
        # 4 bytes command:
        #     0x08 is SET_SENSORS_COMMAND
        #     Each bit in the three following bytes are one sensor.
        self._serial.write(struct.pack('BBBB', 0x08, 0x84, 0x01, 0x00))  # GSR and PPG
        self._wait_for_ack()
        print("[WristbandProxy.py] Shimmer: sensor setting, done.")

        # Enable the internal expansion board power
        self._serial.write(struct.pack('BB', 0x5E, 0x01))
        self._wait_for_ack()
        print("[WristbandProxy.py] Shimmer: enable internal expansion board power, done.")

        # Stop any ongoing datastream (stabilize reconnect situation)
        self._serial.write(struct.pack('B', 0x20))
        print("stop command sent, waiting for ACK_COMMAND")
        self._wait_for_ack()

        # send the set sampling rate command (prevent some issues, when the shimmer get misconfigured)
        '''
        sampling_freq = 32768 / clock_wait = X Hz
        2 << 14 = 32768
        '''
        sampling_freq = 51.2
        clock_wait = math.ceil((2 << 14) / sampling_freq)

        self._serial.write(struct.pack('<BH', 0x05, clock_wait))
        self._wait_for_ack()

        # Inquiry configurations (For finding channels order)
        # Page 16 of This PDF:
        # http://www.shimmersensing.com/images/uploads/docs/LogAndStream_for_Shimmer3_Firmware_User_Manual_rev0.11a.pdf

        self._serial.write(struct.pack('B', 0x01))
        self._wait_for_ack()
        inquiery_response = bytes("", 'utf-8')
        # response_size is 1 packet_type + 2 Sampling rate + 4 Config Bytes +
        # 1 Num Channels + 1 Buffer size
        response_size = 9
        numbytes = 0
        while numbytes < response_size:
            inquiery_response += self._serial.read(response_size)
            numbytes = len(inquiery_response)

        num_channels = inquiery_response[7]
        print("[WristbandProxy.py] Shimmer: Number of Channels:", num_channels)

        if num_channels != 5:
            print("[WristbandProxy.py] There is a problem with the number of channels, it should be 5")

        print("[WristbandProxy.py] Shimmer: Buffer size:", inquiery_response[8])

        # There's one byte for each channel
        # For the meaning of each byte, refer to the above PDF
        channels = bytes("", "utf-8")
        numbytes = 0
        while numbytes < num_channels:
            channels += self._serial.read(num_channels)
            numbytes = len(channels)

        print("[WristbandProxy.py] Shimmer Channel 1:", channels[0])
        print("[WristbandProxy.py] Shimmer Channel 2:", channels[1])
        print("[WristbandProxy.py] Shimmer Channel 3:", channels[2])
        print("[WristbandProxy.py] Shimmer Channel 4:", channels[3])
        print("[WristbandProxy.py] Shimmer Channel 5:", channels[4])

        # send start streaming command
        self._serial.write(struct.pack('B', 0x07))
        self._wait_for_ack()
        print("[WristbandProxy.py] Shimmer: start command sending, done.")

        # start thread to service client socket
        self.t = Thread(target=self._socketServiceTask,) 
        self.t.daemon = True
        self.t.start()
        
        return True

    def _wait_for_ack(self):
        ddata = ""
        ack = struct.pack('B', 0xff)
        while ddata != ack:
            ddata = self._serial.read(1)

    def _stop_shimmer(self):
        # send stop streaming command
        self._serial.write(struct.pack('B', 0x20))

        print("stop command sent, waiting for ACK_COMMAND")
        self._wait_for_ack()
        print("ACK_COMMAND received.")
        self._serial.close()
        print("All done")

    def _socketServiceTask(self):

        self._break_loop = False

        '''
        Reads incoming data
        '''
        ddata = bytes("", 'utf-8')
        numbytes = 0
        # 1byte packet type + 3byte timestamp + 2 byte X + 2 byte Y +
        # 2 byte Z + 2 byte PPG + 2 byte GSR
        framesize = 14

        #try:
        while True:
            while numbytes < framesize:
                ddata += self._serial.read(framesize)
                numbytes = len(ddata)
                if self._break_loop:
                    break

            if self._break_loop:
                self._stop_shimmer()
                break

            data = ddata[0:framesize]
            ddata = ddata[framesize:]
            numbytes = len(ddata)

            # read basic packet information
            (packettype) = struct.unpack('B', data[0:1])
            (timestamp0, timestamp1, timestamp2) = struct.unpack('BBB', data[1:4])

            # read packet payload
            (x, y, z, PPG_raw, GSR_raw) = struct.unpack('HHHHH', data[4:framesize])
            record_time = datetime.datetime.now()

            # get current GSR range resistor value
            data_range = ((GSR_raw >> 14) & 0xff)  # upper two bits
            if data_range == 0:
                rf = 40.2  # kOhm
            elif data_range == 1:
                rf = 287.0  # kOhm
            elif data_range == 2:
                rf = 1000.0  # kOhm
            elif data_range == 3:
                rf = 3300.0  # kOhm

            # convert GSR to kOhm value
            gsr_to_volts = (GSR_raw & 0x3fff) * (3.0 / 4095.0)
            GSR_ohm = rf / ((gsr_to_volts / 0.5) - 1.0)

            # convert PPG to milliVolt value
            PPG_mv = PPG_raw * (3000.0 / 4095.0)

            timestamp = timestamp0 + timestamp1 * 256 + timestamp2 * 65536

            row = [round(time() * 1000),
                   x, y, z,
                   GSR_ohm,
                   PPG_mv]

            #row = [packettype[0],
            #       timestamp,
            #       x, y, z,
            #       GSR_ohm,
            #       PPG_mv,
            #       record_time]

            if self.recording:
                self.adapter.recvData(row)

        #except:
        #    self._stop_shimmer()

    def onEvent(self, event, **data):
        if event == "RECORDING":
            if data is None:
                raise ValueError("malformed event detected, no data associated")
            if "state" not in data:
                raise ValueError("malformed event detected, no state is given")

            self.recording = data["state"]

        pass

    """
        Name: send
        Descr: not suported, yet
    """
    def send(self):
        print("Send not supported")
        pass

    def disconnect(self):
        pass
