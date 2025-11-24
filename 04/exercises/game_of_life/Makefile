build:
	nvcc --std=c++11 game_of_life.cu -Iinclude -Iextern/raylib/include extern/raylib/lib/libraylib.a -o game_of_life

run: build
	./game_of_life