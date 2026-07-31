/**
 * @file t4_char.hpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief 
 * @version 0.1
 * @date 2024-02-01
 * 
 *
 * @copyright GPL3
 * 
 */

#ifndef HDC_T4CHAR_HPP
#define HDC_T4CHAR_HPP

namespace hd {
    struct t4_ints;

    #pragma pack(push,1)
    union t4_char { ///< 4 Bipolar-packed char
        struct {
            int t0:2;
            int t1:2;
            int t2:2;
            int t3:2;
            };
        unsigned char ch;

        t4_char &operator*=(t4_char const& rhs){
            this->t0 *= rhs.t0;
            this->t1 *= rhs.t1;
            this->t2 *= rhs.t2;
            this->t3 *= rhs.t3;
            return *this;
        }

        constexpr t4_char(const int i = 0):
        t0(i),t1(i),t2(i),t3(i)
        {}

        constexpr int operator[](int const& i) const {
            const char a = (this->ch >> (i*2)) & 1;
            const char s = (this->ch >> (i*2 + 1)) & 1;
            return a - (s*2);
        }

    };
    #pragma pack(pop)

    inline bool operator==(t4_char const& lhs, t4_char const& rhs){
        return lhs.ch == rhs.ch;
    }

    inline t4_char operator*(t4_char const& lhs, t4_char const& rhs){
        t4_char ret = lhs;
        ret *= rhs;
        return ret;
    }

    struct t4_ints{
        int t0 = 0;
        int t1 = 0;
        int t2 = 0;
        int t3 = 0;

        t4_ints &operator=(t4_char &rhs) {
            this->t0 = rhs.t0;
            this->t1 = rhs.t1;
            this->t2 = rhs.t2;
            this->t3 = rhs.t3;
            return *this;
        }

        t4_ints(t4_char const& rhs){
            this->t0 = rhs.t0;
            this->t1 = rhs.t1;
            this->t2 = rhs.t2;
            this->t3 = rhs.t3;
        }

        t4_ints(const int rhs = 0){
            this->t0 = rhs;
            this->t1 = rhs;
            this->t2 = rhs;
            this->t3 = rhs;
        }

        t4_ints &operator+=(t4_char const& rhs){
            this->t0 += rhs.t0;
            this->t1 += rhs.t1;
            this->t2 += rhs.t2;
            this->t3 += rhs.t3;
            return *this;
        }

        /**
         * @brief Normalize Operator
         * A normalize function that works inside a SYCL kernel
         * @return t4_char 
         */
        constexpr t4_char normalize(int const& level = 0){
            t4_char ret;
            ret.t0 = (this->t0 > level)-(this->t0 < -level);
            ret.t1 = (this->t1 > level)-(this->t1 < -level);
            ret.t2 = (this->t2 > level)-(this->t2 < -level);
            ret.t3 = (this->t3 > level)-(this->t3 < -level);
            return ret;
        }
    };

}
#endif //HDC_T4CHAR_HPP