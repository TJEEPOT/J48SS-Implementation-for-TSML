package weka.classifiers.trees.j48SS.jmetal;

import java.util.Random;

public class PseudoRandom {
    private static       RandomGenerator random_           = null;
    private static final RandomGenerator defaultGenerator_ = new RandomGenerator();
    
    private PseudoRandom() {
        if (random_ == null) {
            random_ = new RandomGenerator();
        }
    }
    
    public static void setRandomGenerator(int seed) {
        random_ = new MersenneTwisterFast(seed);
    }
    
    public static int randInt() {
        if (random_ == null) {
            random_ = defaultGenerator_;
        }
        
        return random_.nextInt(Integer.MAX_VALUE);
    }
    
    public static double randDouble() {
        if (random_ == null) {
            random_ = defaultGenerator_;
        }
        
        return random_.nextDouble();
    }
    
    public static int randInt(int minBound, int maxBound) {
        if (random_ == null) {
            random_ = defaultGenerator_;
        }
        
        return minBound + random_.nextInt(maxBound - minBound);
    }
    
    private static class RandomGenerator {
        double   seed    = (new Random(System.nanoTime())).nextDouble();
        double[] oldrand = new double[55];
        int      jrand;
        
        public RandomGenerator() {
            this.randomize();
        }
        
        void randomize() {
            for (int j1 = 0; j1 <= 54; ++ j1) {
                this.oldrand[j1] = 0.0D;
            }
            
            this.jrand = 0;
            this.warmup_random(this.seed);
        }
        
        void warmup_random(double seed) {
            this.oldrand[54] = seed;
            double new_random  = 1.0E-9D;
            double prev_random = seed;
            
            for (int j1 = 1; j1 <= 54; ++ j1) {
                int ii = 21 * j1 % 54;
                this.oldrand[ii] = new_random;
                                   new_random = prev_random - new_random;
                if (new_random < 0.0D) {
                    ++ new_random;
                }
                
                prev_random = this.oldrand[ii];
            }
            
            this.advance_random();
            this.advance_random();
            this.advance_random();
            this.jrand = 0;
        }
        
        void advance_random() {
            int    j1;
            double new_random;
            for (j1 = 0; j1 < 24; ++ j1) {
                new_random = this.oldrand[j1] - this.oldrand[j1 + 31];
                if (new_random < 0.0D) {
                    ++ new_random;
                }
                
                this.oldrand[j1] = new_random;
            }
            
            for (j1 = 24; j1 < 55; ++ j1) {
                new_random = this.oldrand[j1] - this.oldrand[j1 - 24];
                if (new_random < 0.0D) {
                    ++ new_random;
                }
                
                this.oldrand[j1] = new_random;
            }
            
        }
        
        double randomperc() {
            ++ this.jrand;
            if (this.jrand >= 55) {
                this.jrand = 1;
                this.advance_random();
            }
            
            return this.oldrand[this.jrand];
        }
        
        public synchronized int nextInt(int upperBound) {
            return this.rndInt(0, upperBound);
        }
        
        public synchronized double nextDouble() {
            return this.randomperc();
        }
        
        public synchronized int rndInt(int low, int high) {
            int res;
            if (low >= high) {
                res = low;
            }
            else {
                res = low + (int)(this.randomperc() * (double)(high - low + 1));
                if (res > high) {
                    res = high;
                }
            }
            
            return res;
        }
    }
    
    /**
     * <h3>MersenneTwister and MersenneTwisterFast</h3>
     * <p><b>Version 16</b>, based on version MT199937(99/10/29)
     * of the Mersenne Twister algorithm found at
     * <a href="http://www.math.keio.ac.jp/matumoto/emt.html">
     * The Mersenne Twister Home Page</a>, with the initialization
     * improved using the new 2002/1/26 initialization algorithm
     * By Sean Luke, October 2004.
     *
     * <p><b>MersenneTwister</b> is a drop-in subclass replacement
     * for java.util.Random.  It is properly synchronized and
     * can be used in a multithreaded environment.  On modern VMs such
     * as HotSpot, it is approximately 1/3 slower than java.util.Random.
     *
     * <p><b>MersenneTwisterFast</b> is not a subclass of java.util.Random.  It has
     * the same public methods as Random does, however, and it is
     * algorithmically identical to MersenneTwister.  MersenneTwisterFast
     * has hard-code inlined all of its methods directly, and made all of them
     * final (well, the ones of consequence anyway).  Further, these
     * methods are <i>not</i> synchronized, so the same MersenneTwisterFast
     * instance cannot be shared by multiple threads.  But all this helps
     * MersenneTwisterFast achieve well over twice the speed of MersenneTwister.
     * java.util.Random is about 1/3 slower than MersenneTwisterFast.
     *
     * <h3>About the Mersenne Twister</h3>
     * <p>This is a Java version of the C-program for MT19937: Integer version.
     * The MT19937 algorithm was created by Makoto Matsumoto and Takuji Nishimura,
     * who ask: "When you use this, send an email to: matumoto@math.keio.ac.jp
     * with an appropriate reference to your work".  Indicate that this
     * is a translation of their algorithm into Java.
     *
     * <p><b>Reference. </b>
     * Makato Matsumoto and Takuji Nishimura,
     * "Mersenne Twister: A 623-Dimensionally Equidistributed Uniform
     * Pseudo-Random Number Generator",
     * <i>ACM Transactions on Modeling and. Computer Simulation,</i>
     * Vol. 8, No. 1, January 1998, pp 3--30.
     *
     * <h3>About this Version</h3>
     *
     * <p><b>Changes Since V15:</b> Added serialVersionUID to quiet compiler warnings
     * from Sun's overly verbose compilers as of JDK 1.5.
     *
     * <p><b>Changes Since V14:</b> made strictfp, with StrictMath.log and StrictMath.sqrt
     * in nextGaussian instead of Math.log and Math.sqrt.  This is largely just to be safe,
     * as it presently makes no difference in the speed, correctness, or results of the
     * algorithm.
     *
     * <p><b>Changes Since V13:</b> clone() method CloneNotSupportedException removed.
     *
     * <p><b>Changes Since V12:</b> clone() method added.
     *
     * <p><b>Changes Since V11:</b> stateEquals(...) method added.  MersenneTwisterFast
     * is equal to other MersenneTwisterFasts with identical state; likewise
     * MersenneTwister is equal to other MersenneTwister with identical state.
     * This isn't equals(...) because that requires a contract of immutability
     * to compare by value.
     *
     * <p><b>Changes Since V10:</b> A documentation error suggested that
     * setSeed(int[]) required an int[] array 624 long.  In fact, the array
     * can be any non-zero length.  The new version also checks for this fact.
     *
     * <p><b>Changes Since V9:</b> readState(stream) and writeState(stream)
     * provided.
     *
     * <p><b>Changes Since V8:</b> setSeed(int) was only using the first 28 bits
     * of the seed; it should have been 32 bits.  For small-number seeds the
     * behavior is identical.
     *
     * <p><b>Changes Since V7:</b> A documentation error in MersenneTwisterFast
     * (but not MersenneTwister) stated that nextDouble selects uniformly from
     * the full-open interval [0,1].  It does not.  nextDouble's contract is
     * identical across MersenneTwisterFast, MersenneTwister, and java.util.Random,
     * namely, selection in the half-open interval [0,1).  That is, 1.0 should
     * not be returned.  A similar contract exists in nextFloat.
     *
     * <p><b>Changes Since V6:</b> License has changed from LGPL to BSD.
     * New timing information to compare against
     * java.util.Random.  Recent versions of HotSpot have helped Random increase
     * in speed to the point where it is faster than MersenneTwister but slower
     * than MersenneTwisterFast (which should be the case, as it's a less complex
     * algorithm but is synchronized).
     *
     * <p><b>Changes Since V5:</b> New empty constructor made to work the same
     * as java.util.Random -- namely, it seeds based on the current time in
     * milliseconds.
     *
     * <p><b>Changes Since V4:</b> New initialization algorithms. See
     * <a href="http://www.math.keio.ac.jp/matumoto/MT2002/emt19937ar.html">Here</a>.
     *
     * <p>The MersenneTwister code is based on standard MT19937 C/C++
     * code by Takuji Nishimura,
     * with suggestions from Topher Cooper and Marc Rieffel, July 1997.
     * The code was originally translated into Java by Michael Lecuyer,
     * January 1999, and the original code is Copyright (c) 1999 by Michael Lecuyer.
     *
     * <h3>Java notes</h3>
     *
     * <p>This implementation implements the bug fixes made
     * in Java 1.2's version of Random, which means it can be used with
     * earlier versions of Java.  See
     * <a href="http://www.javasoft.com/products/jdk/1.2/docs/api/java/util/Random.html">Here</a>
     * the JDK 1.2 java.util.Random documentation for further documentation
     * on the random-number generation contracts made.  Additionally, there's
     * an undocumented bug in the JDK java.util.Random.nextBytes() method,
     * which this code fixes.
     *
     * <p> Just like java.util.Random, this
     * generator accepts a long seed but doesn't use all of it.  java.util.Random
     * uses 48 bits.  The Mersenne Twister instead uses 32 bits (int size).
     * So it's best if your seed does not exceed the int range.
     *
     * <p>MersenneTwister can be used reliably
     * on JDK version 1.1.5 or above.  Earlier Java versions have serious bugs in
     * java.util.Random; only MersenneTwisterFast (and not MersenneTwister nor
     * java.util.Random) should be used with them.
     *
     * <h3>License</h3>
     *
     * Copyright (c) 2003 by Sean Luke. <br>
     * Portions copyright (c) 1993 by Michael Lecuyer. <br>
     * All rights reserved. <br>
     *
     * <p>Redistribution and use in source and binary forms, with or without
     * modification, are permitted provided that the following conditions are met:
     * <ul>
     * <li> Redistributions of source code must retain the above copyright notice,
     * this list of conditions and the following disclaimer.
     * <li> Redistributions in binary form must reproduce the above copyright notice,
     * this list of conditions and the following disclaimer in the documentation
     * and/or other materials provided with the distribution.
     * <li> Neither the name of the copyright owners, their employers, nor the
     * names of its contributors may be used to endorse or promote products
     * derived from this software without specific prior written permission.
     * </ul>
     * <p>THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
     * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
     * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
     * DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNERS OR CONTRIBUTORS BE
     * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
     * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
     * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
     * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
     * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
     * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
     * POSSIBILITY OF SUCH DAMAGE.
     *
     @version 16
     */
    
    // Note: this class is hard-inlined in all of its methods.  This makes some of
    // the methods well-nigh unreadable in their complexity.  In fact, the Mersenne
    // Twister is fairly easy code to understand: if you're trying to get a handle
    // on the code, I strongly suggest looking at MersenneTwister.java first.
    // -- Sean
    
    public static strictfp class MersenneTwisterFast extends RandomGenerator {
        // Period parameters
        private static final int N = 624;
        private static final int M = 397;
        private static final int MATRIX_A = 0x9908b0df;   //    private static final * constant vector a
        private static final int UPPER_MASK = 0x80000000; // most significant w-r bits
        private static final int LOWER_MASK = 0x7fffffff; // least significant r bits
        
        
        // Tempering parameters
        private static final int TEMPERING_MASK_B = 0x9d2c5680;
        private static final int TEMPERING_MASK_C = 0xefc60000;
        
        private int[] mt; // the array for the state vector
        private int   mti; // mti==N+1 means mt[N] is not initialized
        private int[] mag01;
        
        /* We're overriding all internal data, to my knowledge, so this should be okay */
        public Object clone()
        {
            try
            {
                MersenneTwisterFast f = (MersenneTwisterFast)(super.clone());
                f.mt = mt.clone();
                f.mag01 = mag01.clone();
                return f;
            }
            catch (CloneNotSupportedException e) { throw new InternalError(); } // should never happen
        }
        
        
        /**
         * Constructor using a given seed.  Though you pass this seed in
         * as a long, it's best to make sure it's actually an integer.
         *
         */
        public MersenneTwisterFast(final long seed)
        {
            setSeed(seed);
        }
        
        
        /**
         * Initalize the pseudo random number generator.  Don't
         * pass in a long that's bigger than an int (Mersenne Twister
         * only uses the first 32 bits for its seed).
         */
        
        synchronized public void setSeed(final long seed)
        {
            mt = new int[N];
            
                       mag01 = new int[2];
            mag01[0] = 0x0;
            mag01[1] = MATRIX_A;
            
            mt[0]= (int)(seed & 0xffffffff);
            for (mti=1; mti<N; mti++)
            {
                mt[mti] =
                        (1812433253 * (mt[mti-1] ^ (mt[mti-1] >>> 30)) + mti);
                /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
                /* In the previous versions, MSBs of the seed affect   */
                /* only MSBs of the array mt[].                        */
                /* 2002/01/09 modified by Makoto Matsumoto             */
                mt[mti] &= 0xffffffff;
                /* for >32 bit machines */
            }
        }
        
        /** Returns a random double in the half-open range from [0.0,1.0).  Thus 0.0 is a valid
         result but 1.0 is not. */
        public final double nextDouble()
        {
            int y;
            int z;
            
            if (mti >= N)   // generate N words at one time
            {
                int kk;
                final int[] mt = this.mt; // locals are slightly faster
                final int[] mag01 = this.mag01; // locals are slightly faster
                
                for (kk = 0; kk < N - M; kk++)
                {
                    y = (mt[kk] & UPPER_MASK) | (mt[kk+1] & LOWER_MASK);
                    mt[kk] = mt[kk+M] ^ (y >>> 1) ^ mag01[y & 0x1];
                }
                for (; kk < N-1; kk++)
                {
                    y = (mt[kk] & UPPER_MASK) | (mt[kk+1] & LOWER_MASK);
                    mt[kk] = mt[kk+(M-N)] ^ (y >>> 1) ^ mag01[y & 0x1];
                }
                y = (mt[N-1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
                mt[N-1] = mt[M-1] ^ (y >>> 1) ^ mag01[y & 0x1];
                
                mti = 0;
            }
            
            y = mt[mti++];
            y ^= y >>> 11;                          // TEMPERING_SHIFT_U(y)
            y ^= (y << 7) & TEMPERING_MASK_B;       // TEMPERING_SHIFT_S(y)
            y ^= (y << 15) & TEMPERING_MASK_C;      // TEMPERING_SHIFT_T(y)
            y ^= (y >>> 18);                        // TEMPERING_SHIFT_L(y)
            
            if (mti >= N)   // generate N words at one time
            {
                int kk;
                final int[] mt = this.mt; // locals are slightly faster
                final int[] mag01 = this.mag01; // locals are slightly faster
                
                for (kk = 0; kk < N - M; kk++)
                {
                    z = (mt[kk] & UPPER_MASK) | (mt[kk+1] & LOWER_MASK);
                    mt[kk] = mt[kk+M] ^ (z >>> 1) ^ mag01[z & 0x1];
                }
                for (; kk < N-1; kk++)
                {
                    z = (mt[kk] & UPPER_MASK) | (mt[kk+1] & LOWER_MASK);
                    mt[kk] = mt[kk+(M-N)] ^ (z >>> 1) ^ mag01[z & 0x1];
                }
                z = (mt[N-1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
                mt[N-1] = mt[M-1] ^ (z >>> 1) ^ mag01[z & 0x1];
                
                mti = 0;
            }
            
            z = mt[mti++];
            z ^= z >>> 11;                          // TEMPERING_SHIFT_U(z)
            z ^= (z << 7) & TEMPERING_MASK_B;       // TEMPERING_SHIFT_S(z)
            z ^= (z << 15) & TEMPERING_MASK_C;      // TEMPERING_SHIFT_T(z)
            z ^= (z >>> 18);                        // TEMPERING_SHIFT_L(z)
            
            /* derived from nextDouble documentation in jdk 1.2 docs, see top */
            return ((((long)(y >>> 6)) << 27) + (z >>> 5)) / (double)(1L << 53);
        }
        
        
        /** Returns an integer drawn uniformly from 0 to n-1.  Suffice it to say,
         n must be greater than 0, or an IllegalArgumentException is raised. */
        public final int nextInt(final int n)
        {
            if (n<=0)
                throw new IllegalArgumentException("n must be positive, got: " + n);
            
            if ((n & -n) == n)  // i.e., n is a power of 2
            {
                int y;
                
                if (mti >= N)   // generate N words at one time
                {
                    int kk;
                    final int[] mt = this.mt; // locals are slightly faster
                    final int[] mag01 = this.mag01; // locals are slightly faster
                    
                    for (kk = 0; kk < N - M; kk++)
                    {
                        y = (mt[kk] & UPPER_MASK) | (mt[kk+1] & LOWER_MASK);
                        mt[kk] = mt[kk+M] ^ (y >>> 1) ^ mag01[y & 0x1];
                    }
                    for (; kk < N-1; kk++)
                    {
                        y = (mt[kk] & UPPER_MASK) | (mt[kk+1] & LOWER_MASK);
                        mt[kk] = mt[kk+(M-N)] ^ (y >>> 1) ^ mag01[y & 0x1];
                    }
                    y = (mt[N-1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
                    mt[N-1] = mt[M-1] ^ (y >>> 1) ^ mag01[y & 0x1];
                    
                    mti = 0;
                }
                
                y = mt[mti++];
                y ^= y >>> 11;                          // TEMPERING_SHIFT_U(y)
                y ^= (y << 7) & TEMPERING_MASK_B;       // TEMPERING_SHIFT_S(y)
                y ^= (y << 15) & TEMPERING_MASK_C;      // TEMPERING_SHIFT_T(y)
                y ^= (y >>> 18);                        // TEMPERING_SHIFT_L(y)
                
                return (int)((n * (long) (y >>> 1) ) >> 31);
            }
            
            int bits, val;
            do
            {
                int y;
                
                if (mti >= N)   // generate N words at one time
                {
                    int kk;
                    final int[] mt = this.mt; // locals are slightly faster
                    final int[] mag01 = this.mag01; // locals are slightly faster
                    
                    for (kk = 0; kk < N - M; kk++)
                    {
                        y = (mt[kk] & UPPER_MASK) | (mt[kk+1] & LOWER_MASK);
                        mt[kk] = mt[kk+M] ^ (y >>> 1) ^ mag01[y & 0x1];
                    }
                    for (; kk < N-1; kk++)
                    {
                        y = (mt[kk] & UPPER_MASK) | (mt[kk+1] & LOWER_MASK);
                        mt[kk] = mt[kk+(M-N)] ^ (y >>> 1) ^ mag01[y & 0x1];
                    }
                    y = (mt[N-1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
                    mt[N-1] = mt[M-1] ^ (y >>> 1) ^ mag01[y & 0x1];
                    
                    mti = 0;
                }
                
                y = mt[mti++];
                y ^= y >>> 11;                          // TEMPERING_SHIFT_U(y)
                y ^= (y << 7) & TEMPERING_MASK_B;       // TEMPERING_SHIFT_S(y)
                y ^= (y << 15) & TEMPERING_MASK_C;      // TEMPERING_SHIFT_T(y)
                y ^= (y >>> 18);                        // TEMPERING_SHIFT_L(y)
                
                bits = (y >>> 1);
                val = bits % n;
            } while(bits - val + (n-1) < 0);
            return val;
        }
    }
}