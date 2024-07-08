#include "../dependencies/include/init.hpp"
#include "../dependencies/include/preprocess.hpp"
#include "../dependencies/include/solve.hpp"
#include "../dependencies/include/postprocess.hpp"
#include "../dependencies/include/bc.hpp"

int main()
{	
	get_exec_directory();
	preprocess();
	solve();
	// postprocess("T");
	return 0;
}