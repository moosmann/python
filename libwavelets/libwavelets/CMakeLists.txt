set(Headers
	bio.h
	bio_parameters.h
	wavelet_transform.h
	libwavelets_export.h)

set(Sources
	bio_2d.c
	bio_2dy.c
	bio_3d.c
	bio1.2d.c
	bio1.2dy.c
	bio1.3d.c
	bio13579.2d.c
	bio13579.2dy.c
	bio13579.3d.c
	getfilter.c
	wavelet_dec3.c
	wavelet_rec3.c
	wavelet_transform.c)
	
add_library(libwavelets SHARED ${Headers} ${Sources})
