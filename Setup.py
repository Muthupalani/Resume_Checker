from setuptools import setup, Extension

# Define your C extensions
extensions = [
    Extension(
        'pygame',
        sources=[
            'src/base.c', 'src/cdrom.c', 'src/color.c', 'src/constants.c', 
            'src/display.c', 'src/event.c', 'src/fastevent.c', 'src/key.c', 
            'src/mouse.c', 'src/rect.c', 'src/rwobject.c', 'src/surface.c', 
            'src/surflock.c', 'src/time.c', 'src/joystick.c', 'src/draw.c', 
            'src/image.c', 'src/overlay.c', 'src/transform.c', 'src/mask.c', 
            'src/bufferproxy.c', 'src/pixelarray.c', 'src/_arraysurfarray.c'
        ],
        libraries=['SDL', 'SDL_ttf', 'SDL_image', 'SDL_mixer', 'smpeg', 'png', 'jpeg', 'X11', 'portmidi', 'porttime'],
        extra_compile_args=['-D_REENTRANT'],
        include_dirs=['/usr/include/SDL'],
    )
]

# Setup function
setup(
    name="pygame",
    version="2.1.0",  # Change this as needed
    ext_modules=extensions,
    packages=['pygame'],
    install_requires=['numpy'],  # Add any other Python dependencies here
)
