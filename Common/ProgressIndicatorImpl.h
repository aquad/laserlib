/* ProgressIndicatorImpl.h
 *
 * Copyright (C) 2013 Alastair Quadros.
 *
 * This file is part of LaserLib.
 *
 * LaserLib is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * LaserLib is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
 * more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with LaserLib.  If not, see <http://www.gnu.org/licenses/>.
 *
 * \author     Alastair Quadros
 * \date       01-14-2013
*/

#ifndef PROGRESS_INDICATOR_IMPL_HEADER_GUARD
#define PROGRESS_INDICATOR_IMPL_HEADER_GUARD

#include <iostream>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/bind.hpp>
#include <boost/asio.hpp>


class ProgressIndicatorImpl
{
public:
    ProgressIndicatorImpl(int total, int _printPeriod)
        :   totalCount(total),
            printPeriod(_printPeriod),
            sharedCount(0),
            perc(0),
            started(false),
            dTimer(io, boost::posix_time::seconds(printPeriod))
    {}

    ~ProgressIndicatorImpl() { stop(); }

    //starts a new thread that does the waiting
    void start()
    {
        started = true;
        startTime = boost::posix_time::microsec_clock::local_time();
        dTimer.async_wait( boost::bind(&ProgressIndicatorImpl::timerCallback, this, boost::asio::placeholders::error) );
        progThread = boost::thread(&ProgressIndicatorImpl::run, this);
    }

    void run()
    {
        io.run(); //returns when no jobs remain, or stop is called
    }

    //print the total elapsed time, stop callback timer and join threads.
    void stop()
    {
        if( !started )
            return;

        boost::posix_time::time_duration elapsed = boost::posix_time::microsec_clock::local_time() - startTime;
        double elapsedSec = elapsed.total_milliseconds()/1000.0;
        double secs = int(elapsedSec)%60 + ( elapsedSec - int(elapsedSec) ); //more accurate for total
        int mins = (int(elapsedSec)/60)%60;
        int hrs = int(elapsedSec)/3600;

        std::cout << "Took ";
        if( hrs > 0 )
        {
            std::cout << hrs << "h ";
        }
        if( mins > 0 )
        {
            std::cout << mins << "m ";
        }
        std::cout << secs << "s" << std::endl;

        io.stop();
        progThread.join();
        started = false;
    }

    void timerCallback(const boost::system::error_code &e)
    {
        if( e == boost::asio::error::operation_aborted )
            return;
        if( sharedCount != 0 )
        {
            perc = float(sharedCount)/totalCount*100;
            print();
        }
        dTimer.expires_from_now(boost::posix_time::seconds(printPeriod));
        dTimer.async_wait( boost::bind(&ProgressIndicatorImpl::timerCallback, this, boost::asio::placeholders::error) );
    }

    void print()
    {
        boost::posix_time::time_duration elapsed = boost::posix_time::microsec_clock::local_time() - startTime;
        double elapsedSec = elapsed.total_milliseconds()/1000.0;
        double eta = elapsedSec / perc * (100-perc);
        //std::cout << "elapsed: " << elapsedSec << std::endl;
        int secs = int(eta)%60;
        int mins = (int(eta)/60)%60;
        int hrs = int(eta)/3600;

        std::cout << int(perc) << "%, eta ";
        if( hrs > 0 )
        {
            std::cout << hrs << "h ";
        }
        if( mins > 0 )
        {
            std::cout << mins << "m ";
        }
        std::cout << secs << "s          \r" << std::flush;
    }

    void operator+=(int iters)
    {
        //boost::lock_guard<boost::mutex> lock(countMutex);
        #pragma omp critical
        {
        //countMutex.lock();
            sharedCount += iters;
        }
        //countMutex.unlock();
        //countCondition.notify_one();
    }

    void reset(int totalCount_)
    {
        stop();
        startTime = boost::posix_time::microsec_clock::local_time();
        perc = 0;
        totalCount = totalCount_;
        sharedCount = 0;
        start();
    }

    unsigned int totalCount, sharedCount;
    float perc;
    int printPeriod;
    boost::posix_time::ptime startTime;
    boost::thread progThread;
    boost::condition_variable countCondition;
    boost::mutex countMutex;
    bool started;
    boost::asio::io_service io;
    boost::asio::deadline_timer dTimer;
};


#endif //PROGRESS_INDICATOR_IMPL_HEADER_GUARD
