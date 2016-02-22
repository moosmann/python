
int compimages(
    FLOAT*,  //vector1
    FLOAT*,  //vector2
    FLOAT*,  //diffvector
    int,     //Size
    double*, //ptrmaxvalue
    double*, //ptrposerror
    double*, //ptrnegerror
    double*, //ptrerror2
    double*, //ptrnorm2
    double*, //ptrerror1
    double*, //ptrnorm1
    double*, //ptrerrorlog10
    int*,    //ptrnonexactnumber
    int*,    //ptrsize
    double*  //ptrsum
    );

int show_compresults(double, // maxvalue
                     double, // poserror
                     double, // negerror
                     double, // error2
                     double, // norm2
                     double, // error1
                     double, // norm1
                     double, // errorlog10
                     int,    // nonexactnumber
                     int,    // size
                     FLOAT*, // *SNR_ptr
                     int,    // typesize
                     double  // sum
                     );
