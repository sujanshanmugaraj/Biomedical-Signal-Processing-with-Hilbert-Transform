## HOW HILBERT TRANSFORM IS APPLIED IN THE MAIN PROJECT

1.**Original Signal:** Let's denote the original real-valued signal as  x(t) , where  t  represents time (or space).

2.**Fourier Transform:** The first step in performing the Hilbert transform involves transforming the original signal from the time domain to the frequency domain using the Fourier transform. The Fourier transform of  x(t)  is denoted as  X(f) , where  f  represents frequency.

3.**Frequency Response:** In the frequency domain, each frequency component of the original signal  x(t)  is represented by a complex number  X(f) . This complex number has both magnitude (amplitude) and phase.

4. **Hilbert Filter:** The next step involves applying a special filter, called the Hilbert filter or Hilbert kernel, to the frequency response X(f) . The Hilbert filter is designed such that it has a phase shift of  ±90°  or  ±π/2  radians for all frequencies.

5. **Complex Signal:** After applying the Hilbert filter to X(f) , we obtain a complex-valued function in the frequency domain, denoted as  H(X(f)) . This complex function represents the analytic signal associated with the original real-valued signal  x(t).

**6. Inverse Fourier Transform:** Finally, the inverse Fourier transform is applied to  H(X(f)) to obtain the analytic signal in the time domain. The resulting analytic signal, denoted as  y(t) , is a complex-valued function that contains information about both the amplitude and phase of the original signal  x(t) .




