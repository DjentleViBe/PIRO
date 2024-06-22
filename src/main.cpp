#include "../dependencies/include/init.hpp"
#include "../dependencies/include/preprocess.hpp"
#include "../dependencies/include/solve.hpp"
#include "../dependencies/include/postprocess.hpp"

int main()
{	
	init();
	preprocess();
	solve();
	postprocess();
	return 0;
}