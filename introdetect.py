import glob
import cv2
import os
import collections
import numpy as np
import datetime
import math

folder = r"P:/Serien/Alf/"

class Settings(): 
    intro_starttime_min = 0
    intro_endtime_max = 300 
    intro_duration_max = 90 
    intro_duration_min = 10 
    intro_duration_filtered = intro_duration_min
    
    deviation_infinity = 100000 #=const
    deviation_factor = 1.3 #which image deviation is allowed to be still within the intro
    deviation_average = 4 #average over 4 frames
    deviation_required = 24 # might be video source dependent?
    deviation_accepted = 0 #  set to 0 to deactivate, will learn with filter_tau
     
    
    dt_rough = 0.7 #rough range of similar images between movies
    dt_detailed = 0.04 #for matching two movies in a second more detailed step 0.04 = 25FPS    
    dt_output_resolution = int(1.0 / deviation_average / dt_detailed)*dt_detailed #1sec of output resultion
    dt_rough = int(dt_rough / dt_detailed)*dt_detailed
    
    dualRange = int(intro_duration_min / dt_rough)*dt_rough #time in seconds that is *smaller* than the complete intro
    
    filter_tau = math.exp(-1.0/0.5) #  set to 0 to deactivate, 1=sample time, 0.5=filter time  math.exp(-1.0/0.5) => adapts after ~10 iterations

class Movie():
    
    def __init__(self, filename):
        self.cap = cv2.VideoCapture(filename) # Read single frame avi        
        self.filename = filename
        self.map = {}  #buffers images
        
    #===========================================================================
    # def buffer(self):
    #     if len(self.map)<10:
    #         self.cap.set(cv2.CAP_PROP_POS_MSEC,0)
    #         time = Settings.intro_starttime_min
    #         print("Buffering...")
    #         while time<Settings.intro_endtime_max:        
    #             frame = self.cap.read()
    #             time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    #             self.map[time]  = frame;
    #         print("Finished...")
    #===========================================================================
                
    def getFrameDirect(self, timeStamp):
        self.cap.set(cv2.CAP_PROP_POS_MSEC,timeStamp*1000)
        _, current_frame = self.cap.read()
        return  cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
        
    def getFrame(self, timeStamp):        
        
        if len(self.map)>0: 
            timeStamp2 = min(self.map.keys(), key=lambda k: abs(k-timeStamp))
            if abs(timeStamp2-timeStamp)<Settings.dt_detailed:
                timeStamp = timeStamp2
            
        if timeStamp in self.map:
            return self.map[timeStamp]
        else:
            current_frame = self.getFrameDirect(timeStamp)
            self.map[timeStamp] = current_frame 
            
        return current_frame


class CompareMovies():    
    
    def getDeviation(self, frame1, frame2):
        if frame1.shape != frame2.shape: #scale to same format 
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]), interpolation = cv2.INTER_AREA)
        return cv2.norm(frame1, frame2, cv2.NORM_L1) / (frame1.shape[1]*frame1.shape[0])
    
    
    def getDeviationMovie(self, m1, timeStamp1, m2, timeStamp2):
        frame1 = m1.getFrame(timeStamp1)
        frame2 = m2.getFrame(timeStamp2)                     
        return self.getDeviation(frame1, frame2)     
    
    
    def putFrames(self, m1, m2, t1, t2, dim):
        frame1 = m1.getFrame(t1)
        frame2 = m2.getFrame(t2)
        
        scale1 = cv2.resize(frame1, dim, interpolation = cv2.INTER_AREA)
        scale2 = cv2.resize(frame2, dim, interpolation = cv2.INTER_AREA)
        scaleX = cv2.resize(scale1-scale2, dim, interpolation = cv2.INTER_AREA)
        return np.hstack((scale1, scale2, scaleX))
          
          
    def showFrames(self, m1, m2, t1, t2, bt1, bt2, dev, caption = ""):
        cv2.destroyAllWindows()
        
        frame1 = m1.getFrame(t1)        
        
        scale = 0.6
        width2 = int(frame1.shape[1] * scale)
        height2 = int(frame1.shape[0] * scale)
        dim = (width2, height2) 
        
        final = self.putFrames(m1,m2,t1,t2, dim)
        if bt1 is not None and t1 != bt1:
            finalb = self.putFrames(m1,m2,bt1,bt2, dim)
            final = np.vstack((final,finalb))
                
        title = caption+' '+" Deviation="+str(round(dev))
        final = cv2.cvtColor(final, cv2.COLOR_GRAY2RGB)      
        final = cv2.putText(final, title, (50, 50) , cv2.FONT_HERSHEY_SIMPLEX,  0.6, (191, 255, 0) , 2, cv2.LINE_AA) 
        
        cv2.imshow(title, final)
        cv2.waitKey(1) #update frame
    def findSameImageInternal(self, m1, m2, times1, times2, dualRange, caption, deviation_accepted):
        self.bestResult = (0, 0, Settings.deviation_infinity) 
        
        for time1 in times1:            
            for time2 in times2:
                dev = self.getDeviationMovie(m1, time1, m2, time2)
                
                if dualRange>0:
                    dev = 0.5*(dev + self.getDeviationMovie(m1, time1+dualRange, m2, time2+dualRange)) #calc mean value of deviations
                                
                if  dev < self.bestResult[2]:
                    self.showFrames(m1,m2,time1, time2, time1+dualRange, time2+dualRange, dev, caption)
                    self.bestResult = (time1, time2, dev)
                    
                if dev < deviation_accepted:
                    return self.bestResult
                    
        return self.bestResult
    
    
    def findSameImage(self, m1, m2, t1, t2, minT, maxT, dt, dualRange, templateTime, caption, deviation_accepted):
        if templateTime is None:
            times1 = np.arange(t1 + minT, t1 + maxT, dt)
        else:
            times1 = np.linspace( templateTime[0], templateTime[1], 5)
            times1 = times1[1:-1] 
        times2 = np.arange(t2 + minT, t2 + maxT, dt)        
        return self.findSameImageInternal(m1, m2, times1, times2, dualRange, caption, deviation_accepted)
    
    
    def findIntroRange(self, m1, m2, t1, t2, maxRange, dt, maxDevThreshold, caption):
        if maxRange < 0 and dt > 0:
            dt = -dt
        times = np.arange(0, maxRange, dt)
        maxDev = 0
        lst = collections.deque([], maxlen=Settings.deviation_average)
        for t in times:
            time1 = t1 + t
            time2 = t2 + t
            
            if time1 > 0 and time2 > 0:
                dev = self.getDeviationMovie(m1, time1, m2, time2)
                lst.append(dev)
                
            meanDev = sum(lst) / len(lst)
            if meanDev > maxDev:
                maxDev = meanDev
                    
            #print(t, round(dev), round(meanDev), round(maxDevThreshold), round(maxDev) )
            #self.showFrames(m1,m2,time1, time2, None, None, dev, caption)                            
                 
            if sum(lst) / len(lst) > maxDevThreshold or time1 < 0 or time2 < 0:
                time1 = max(0.0, time1)
                time2 = max(0.0, time2)
                self.showFrames(m1, m2, time1, time2, None, None, dev, caption)
                return (time1, time2, dev)
            
        
        if maxDevThreshold < Settings.deviation_infinity:            
            print( meanDev, maxDev, " is not > ", maxDevThreshold)
            raise Exception("EXCEPTION: Nothing found in findIntroRange")
        else:
            return (None, None, maxDev)

    def analyzeTwoFiles(self, movie1, movie2, templateTime):
        print("------------------------------------")
        print("Analyze 2 Files: ", movie1.filename, movie2.filename)
        
        #find same movie part in a rough way
        (roughTime1, roughTime2, dev) = self.findSameImage(movie1, movie2,          0,          0, Settings.intro_starttime_min, Settings.intro_endtime_max, Settings.dt_rough, Settings.dualRange, templateTime, "Rough", Settings.deviation_accepted)
        print("Rough search: ", roughTime1, roughTime2, dev)
        
        #find same movie part in a detailed way
        (midTime1, midTime2, middev)   = self.findSameImage(movie1, movie2, roughTime1, roughTime2,      -Settings.dt_rough,      +Settings.dt_rough,     Settings.dt_detailed, Settings.dualRange, None, "Detailed", 0)
        print("Detailed search", midTime1, midTime2, middev)
        
        if middev > Settings.deviation_required:
            raise Exception("No good match found"+ str(middev)+' > '+str(Settings.deviation_required))        
        
        (endtime1, endtime2, maxdev)  = self.findIntroRange(movie1, movie2, midTime1, midTime2,   Settings.dualRange,  Settings.dt_output_resolution, Settings.deviation_infinity, "MaxDev")
        print("Get max deviation", endtime1, endtime2, maxdev)
        
        (endtime1, endtime2, enddev)  = self.findIntroRange(movie1, movie2, midTime1, midTime2,   Settings.intro_duration_max,  Settings.dt_output_resolution, maxdev*Settings.deviation_factor, "EndTime" )
        print("Search end time", endtime1, endtime2, enddev)
        
        (starttime1, starttime2, startdev) = self.findIntroRange(movie1, movie2, midTime1, midTime2,  -Settings.intro_duration_max, -Settings.dt_output_resolution, maxdev*Settings.deviation_factor, "StartTime" )
        print("Search start time", starttime1, starttime2, startdev)
        
        duration = endtime1-starttime1
        if duration < max( Settings.dualRange, Settings.intro_duration_filtered * 0.8):
            raise Exception("EXCEPTION: Intro time too short: " + str(starttime2) + " - " + str(endtime2) + "  Duration: " + str(duration) + "  DualRange: " + str(Settings.dualRange) + "  FilteredDuration: " +str(Settings.intro_duration_filtered * 0.8) )
        
        Settings.deviation_accepted = Settings.deviation_accepted * (1-Settings.filter_tau) + dev * Settings.filter_tau
        Settings.intro_duration_filtered = Settings.intro_duration_filtered * (1-Settings.filter_tau) + duration * Settings.filter_tau
        
        print("----------Results--------------")        
        print(movie1.filename, starttime1, endtime1, endtime1-starttime1, Settings.intro_duration_filtered)
        print(movie2.filename, starttime2, endtime2, endtime2-starttime2, Settings.intro_duration_filtered)
        print("------------------------------")
        return (starttime1, starttime2, endtime1, endtime2)
    
    
class EdlFile():
    
    def getEdlFilename(self, filename):
        return os.path.splitext(filename)[0]+'.edl'
    
    
    def readEdlFilename(self, filename):
        edlfile = self.getEdlFilename(filename)
        try:
            with open(edlfile) as f:            
                array = [[float(x) for x in line.split() if len(x)>0] for line in f if not line.startswith('#') and len(line.strip())>0]
                result = (array[0][0], array[0][1]) 
        except FileNotFoundError:
            result = None
        except Exception as e:
            print(e)
            result = None
        
        return result
    
    
    def createEdlFile(self, filename, startTime, endTime):
        outfile = self.getEdlFilename(filename)
        print("Creating edl file", outfile)
        with open(outfile, "w") as output:
            output.write( '## EDL File for ' + filename + '\n')
            output.write( '## Created by ' + __file__ + '\n')
            output.write( '## Created on ' +  str(datetime.datetime.now()) + '\n')
            output.write( '## see EDL details on https://kodi.wiki/view/Edit_decision_list' + '\n')            
            output.write( '## StartTime ' + str(datetime.timedelta(seconds=startTime))+'\n')
            output.write( '## EndTime ' + str(datetime.timedelta(seconds=endTime))+'\n')
            output.write( '## Duration ' + str(endTime-startTime)+'\n')
            output.write( '## ---------------------------------------------------------'+'\n')
            output.write( str(startTime)+' '+str(endTime) + ' 3' ) 


def runSeries(folder):
    files = sorted(glob.glob(folder+"*.*"))
    files = [x for x in files if (x.endswith(".avi") or x.endswith(".mkv") or x.endswith(".mp4") ) and not "youtubeAudio" in x ]    
    print(files)
    
    movies = CompareMovies()
    
    #movies.analyzeTwoFiles(Movie(r"P:\Serien\Alf\Alf_S01E01_16.06.26_11-50_rtlnitro_25_TVOON_DE.mpg.cut.avi"), Movie(r"P:\Serien\Alf\Alf_S01E02_15.11.16_19-50_rtlnitro_25_TVOON_DE.mpg.cut.avi"), None)
    #return 
    
    edl = EdlFile()
    
    templateMovie = Movie(files[0]) 
    templateTime = None
    
    for f in files[1:]:
        movie = Movie(f)
        
        #if edl file already exists, do nothing
        if edl.readEdlFilename(movie.filename) is None or edl.readEdlFilename(templateMovie.filename) is None:
        
            #if edl file already exists, read it out for the template
            if templateTime is None:
                templateTime= edl.readEdlFilename(templateMovie.filename)
                           
            #if edl file did not exists, serach for the intro and create edl file
            if templateTime is None:
                try:
                    print("Try to find template")
                    (starttime1, starttime2, endtime1, endtime2) = movies.analyzeTwoFiles(templateMovie, movie, templateTime)
                    print("Template found")
                    
                    edl.createEdlFile(templateMovie.filename, starttime1, endtime1)
                    edl.createEdlFile(movie.filename, starttime2, endtime2)
                    
                    templateTime = (starttime1, endtime1)                
                except Exception as e:
                    print(e)
                    print("Template NOT found. Go to next two movies and try again")
                    templateTime = None
                    templateMovie = movie                
            else:
                try:
                    (_,startTime,_,endTime) = movies.analyzeTwoFiles(templateMovie, movie, templateTime)                    
                    edl.createEdlFile(movie.filename, startTime, endTime)
                except Exception as e:
                    print(e)
                    templateTime = None
                    templateMovie = movie
                         
        #update template since it is more probable that episodes close to each other have a better similarity)
        if edl.readEdlFilename(movie.filename) is not None:
            templateMovie = movie
            templateTime = edl.readEdlFilename(movie.filename)
   
   
if __name__ == "__main__":        
    runSeries(folder) 
    print("Script finished.")
    cv2.waitKey(0)
