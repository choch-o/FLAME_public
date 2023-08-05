"""
Convient power measurement script for the Jetson TX2/Tegra X2.
"""
import os, pdb, pickle, sys, argparse
import numpy as np
# import tensorflow as tf
from train import train_FL_local

# descr, i2c-addr, channel
_nodes = [('module/main', '0040', '0'),
          ('module/cpu+gpu', '0040', '1'),
          ('module/soc', '0040', '2'),
          ]

_valTypes = ['power', 'voltage', 'current']
_valTypesFull = ['power [mW]', 'voltage [mV]', 'current [mA]']


def getNodes():
    """Returns a list of all power measurement nodes, each a
    tuple of format (name, i2d-addr, channel)"""
    return _nodes


def getNodesByName(nameList=['module/main']):
    return [_nodes[[n[0] for n in _nodes].index(name)] for name in nameList]


def powerSensorsPresent():
    """Check whether we are on the TX2 platform/whether the sensors are present"""
    return os.path.isdir('/sys/bus/i2c/drivers/ina3221x/7-0041/iio_device/')


def getPowerMode():
    return os.popen("nvpmodel -q | grep 'Power Mode'").read()[15:-1]


def readValue(i2cAddr='0041', channel='0', valType='power'):
    """Reads a single value from the sensor"""
    fname = '/sys/bus/i2c/drivers/ina3221x/7-%s/iio_device/in_%s%s_input' % (i2cAddr, valType, channel)
    with open(fname, 'r') as f:
        return f.read()


def getModulePower():
    """Returns the current power consumption of the entire module in mW."""
    return float(readValue(i2cAddr='0040', channel='0', valType='power'))


def getAllValues(nodes=_nodes):
    """Returns all values (power, voltage, current) for a specific set of nodes."""
    return [[float(readValue(i2cAddr=node[1], channel=node[2], valType=valType))
             for valType in _valTypes]
            for node in nodes]


def printFullReport():
    """Prints a full report, i.e. (power,voltage,current) for all measurement nodes."""
    from tabulate import tabulate
    header = []
    header.append('description')
    for vt in _valTypesFull:
        header.append(vt)

    resultTable = []
    for descr, i2dAddr, channel in _nodes:
        row = []
        row.append(descr)
        for valType in _valTypes:
            row.append(readValue(i2cAddr=i2dAddr, channel=channel, valType=valType))
        resultTable.append(row)
    print(tabulate(resultTable, header))


import threading
import time


class PowerLogger:
    """This is an asynchronous power logger.
    Logging can be controlled using start(), stop().
    Special events can be marked using recordEvent().
    Results can be accessed through
    """

    def __init__(self, interval=0.01, nodes=_nodes):
        """Constructs the power logger and sets a sampling interval (default: 0.01s)
        and fixes which nodes are sampled (default: all of them)"""
        self.interval = interval
        self._startTime = -1
        self.eventLog = []
        self.dataLog = []
        self._nodes = nodes

    def start(self):
        "Starts the logging activity"""

        # define the inner function called regularly by the thread to log the data
        def threadFun():
            # start next timer
            self.start()
            # log data
            t = self._getTime() - self._startTime
            self.dataLog.append((t, getAllValues(self._nodes)))
            # ensure long enough sampling interval
            t2 = self._getTime() - self._startTime
            assert (t2 - t < self.interval)

        # setup the timer and launch it
        self._tmr = threading.Timer(self.interval, threadFun)
        self._tmr.start()
        if self._startTime < 0:
            self._startTime = self._getTime()

    def _getTime(self):
        # return time.clock_gettime(time.CLOCK_REALTIME)
        return time.time()

    def recordEvent(self, name):
        """Records a marker a specific event (with name)"""
        t = self._getTime() - self._startTime
        self.eventLog.append((t, name))

    def stop(self):
        """Stops the logging activity"""
        self._tmr.cancel()

    def getDataTrace(self, nodeName='module/main', valType='power'):
        """Return a list of sample values and time stamps for a specific measurement node and type"""
        pwrVals = [itm[1][[n[0] for n in self._nodes].index(nodeName)][_valTypes.index(valType)]
                   for itm in self.dataLog]
        timeVals = [itm[0] for itm in self.dataLog]
        return timeVals, pwrVals

    def showDataTraces(self, names=None, valType='power', showEvents=True):
        """creates a PyPlot figure showing all the measured power traces and event markers"""
        if names == None:
            names = [name for name, _, _ in self._nodes]

        # prepare data to display
        TPs = [self.getDataTrace(nodeName=name, valType=valType) for name in names]
        Ts, _ = TPs[0]
        Ps = [p for _, p in TPs]
        energies = [self.getTotalEnergy(nodeName=nodeName) for nodeName in names]
        Ps = list(map(list, zip(*Ps)))  # transpose list of lists

        # draw figure
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.plot(Ts, Ps)
        plt.xlabel('time [s]')
        plt.ylabel(_valTypesFull[_valTypes.index(valType)])
        plt.grid(True)
        plt.legend(['%s (%.2f J)' % (name, enrgy / 1e3) for name, enrgy in zip(names, energies)])
        plt.title('power trace (NVPModel: %s)' % (os.popen("nvpmodel -q | grep 'Power Mode'").read()[15:-1],))
        if showEvents:
            for t, _ in self.eventLog:
                plt.axvline(x=t, color='black')
        plt.show()

    def showMostCommonPowerValue(self, nodeName='module/main', valType='power', numBins=100):
        """computes a histogram of power values and print most frequent bin"""
        import numpy as np
        _, pwrData = np.array(self.getDataTrace(nodeName=nodeName, valType=valType))
        count, center = np.histogram(pwrData, bins=numBins)
        # import matplotlib.pyplot as plt
        # plt.bar((center[:-1]+center[1:])/2.0, count, align='center')
        maxProbVal = center[np.argmax(count)]  # 0.5*(center[np.argmax(count)] + center[np.argmax(count)+1])
        print('max frequent power bin value [mW]: %f' % (maxProbVal,))

    def getTotalEnergy(self, nodeName='module/main', valType='power', idlePower=0):
        """Integrate the power consumption over time."""
        timeVals, dataVals = self.getDataTrace(nodeName=nodeName, valType=valType)
        assert (len(timeVals) == len(dataVals))
        tPrev, wgtdSum = 0.0, 0.0
        for t, d in zip(timeVals, dataVals):
            wgtdSum += (d - idlePower) * (t - tPrev)
            tPrev = t
        return wgtdSum

    def getAveragePower(self, nodeName='module/main', valType='power'):
        energy = self.getTotalEnergy(nodeName=nodeName, valType=valType)
        timeVals, _ = self.getDataTrace(nodeName=nodeName, valType=valType)
        return energy / timeVals[-1]


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Power profiling params')
    parser.add_argument('--gpu', dest='gpu_train', action='store_true')
    parser.add_argument('--no-gpu', dest='gpu_train', action='store_false')
    parser.set_defaults(gpu_train=False)
    parser.add_argument('--outfile', type=str, required=True)
    parser.add_argument('--epochs', type=int)
    args = parser.parse_args()
    
    if args.gpu_train:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else: 
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # printFullReport()

    avg_idle_power =  0
    avg_epoch_times = []
    avg_energy_consumption = 0
    avg_fl_power = 0

    num_times = 5.0

    for i in range(0, int(num_times)):

        if i == 0:
            pl = PowerLogger(interval=0.3, nodes=list(filter(lambda n: n[0].startswith('module/'), getNodes())))
            pl.start()
            pl.recordEvent('idle power calculation')

            time.sleep(15)

            pl.stop()

            idle_power = pl.getAveragePower()

        # pdb.set_trace()


        pl = PowerLogger(interval=0.3, nodes=list(filter(lambda n: n[0].startswith('module/'), getNodes())))
        pl.start()
        pl.recordEvent('run model!')

        epoch_times = train_FL_local(args.epochs)

        time.sleep(1)
        pl.stop()

        
        
        # pl.getTotalEnergy(nodeName='module/main') - (pl.getTotalEnergy(nodeName='module/cpu+gpu') + pl.getTotalEnergy(nodeName='module/soc'))


        energy_consumption = pl.getTotalEnergy(idlePower=idle_power)
        fl_power = pl.getAveragePower()

        avg_idle_power += idle_power
        if len(avg_epoch_times) == 0:
            avg_epoch_times = epoch_times
        else:
            avg_epoch_times = [sum(x) for x in zip(avg_epoch_times, epoch_times)]
        avg_fl_power += fl_power
        avg_energy_consumption += energy_consumption

        time.sleep(30)

    output_dict = {}

    output_dict['total_energy_consumption'] = avg_energy_consumption/num_times
    output_dict['avg_idle_power'] = avg_idle_power/num_times
    output_dict['avg_fl_power'] = avg_fl_power/num_times
    output_dict['epoch_times'] = [time/num_times for time in avg_epoch_times]

    f = open(args.outfile + '.pickle', "wb")
    pickle.dump(output_dict, f)
    f.close()

    # pdb.set_trace()
    # pl.showDataTraces()