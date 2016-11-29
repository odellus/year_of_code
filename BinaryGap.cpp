#include <string>
#include <bitset>

int solution(int N) {

    std::string binary = std::bitset<31>(N).to_string();
    int L = 31;
    int leading = 0;
    for (int i=0; i < L; i++){
        char ch = binary[i];
        if (ch == '1'){
            leading = i;
            break;}
    }
    int tmp_gap = 0;
    int max_gap = 0;
    bool last_was_zero = false;
    for (int i = leading; i < L; i++){
        char ch = binary[i];
        if (!last_was_zero){
            if (ch == '0') {
                tmp_gap = 1;
                last_was_zero = true;
            }
            else{
                tmp_gap = 0;
            }
        }
        else{
            if (ch == '0'){
                tmp_gap += 1;
            }
            else{
                if (tmp_gap > max_gap){
                    max_gap = tmp_gap;
                    }
                last_was_zero = false;
            }
        }
    }

    return max_gap;
}
