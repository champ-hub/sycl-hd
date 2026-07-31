/**
 * @file b8_char.hpp
 * @author Pedro Caires (josepedrocaires@tecnico.ulisboa.pt)
 * @brief 
 * @version 0.1
 * @date 2024-02-01
 * 
 * TODO: operator overloads
 *
 * @copyright GPL3
 * 
 */

#ifndef HDC_B8CHAR_HPP
#define HDC_B8CHAR_HPP

#include <ostream>
#include <sys/types.h>

namespace hd {
    
    #pragma pack(push,1)
    union b8_char { /// 8 Bit-packed char
        struct {
            uint b0:1;
            uint b1:1;
            uint b2:1;
            uint b3:1;
            uint b4:1;
            uint b5:1;
            uint b6:1;
            uint b7:1;
        };
        unsigned char ch;

        b8_char(const int i = 0) : b0(i),b1(i),b2(i),b3(i),b4(i),b5(i),b6(i),b7(i){}

        b8_char(const int ib0,
                const int ib1,
                const int ib2,
                const int ib3,
                const int ib4,
                const int ib5,
                const int ib6,
                const int ib7
        ) : b0(ib0),b1(ib1),b2(ib2),b3(ib3),
            b4(ib4),b5(ib5),b6(ib6),b7(ib7){}
        
        constexpr bool operator[](int i) const {
            return (this->ch >> i) & 1;
        }

        b8_char &operator*=(b8_char const& rhs){
            this->ch ^= rhs.ch;
            return *this;
        }

    };
    #pragma pack(pop)

    struct b8_ints{
        int b0;
        int b1;
        int b2;
        int b3;
        int b4;
        int b5;
        int b6;
        int b7;

        b8_ints(int const i = 0) : b0(i), b1(i), b2(i), b3(i), b4(i), b5(i), b6(i), b7(i) {};

        b8_ints(b8_char const& rhs):
            b0((rhs.b0*2)-1),
            b1((rhs.b1*2)-1),
            b2((rhs.b2*2)-1),
            b3((rhs.b3*2)-1),
            b4((rhs.b4*2)-1),
            b5((rhs.b5*2)-1),
            b6((rhs.b6*2)-1),
            b7((rhs.b7*2)-1)
        {}

        b8_ints &operator=(b8_char&& rhs) {
            this->b0 = (rhs.b0*2)-1;
            this->b1 = (rhs.b1*2)-1;
            this->b2 = (rhs.b2*2)-1;
            this->b3 = (rhs.b3*2)-1;
            this->b4 = (rhs.b4*2)-1;
            this->b5 = (rhs.b5*2)-1;
            this->b6 = (rhs.b6*2)-1;
            this->b7 = (rhs.b7*2)-1;
            return *this;
        }

        b8_ints &operator=(b8_char& rhs) {
            *this = b8_ints(rhs);
            return *this;
        }

        b8_ints &operator+=(b8_ints&& rhs) {
            this->b0 += rhs.b0;
            this->b1 += rhs.b1;
            this->b2 += rhs.b2;
            this->b3 += rhs.b3;
            this->b4 += rhs.b4;
            this->b5 += rhs.b5;
            this->b6 += rhs.b6;
            this->b7 += rhs.b7;
            return *this;
        }

        b8_ints &operator+=(b8_char& rhs) {
            this->b0 += (rhs.b0*2)-1;
            this->b1 += (rhs.b1*2)-1;
            this->b2 += (rhs.b2*2)-1;
            this->b3 += (rhs.b3*2)-1;
            this->b4 += (rhs.b4*2)-1;
            this->b5 += (rhs.b5*2)-1;
            this->b6 += (rhs.b6*2)-1;
            this->b7 += (rhs.b7*2)-1;
            return *this;
        }

    };


    inline b8_char operator*(b8_char const& lhs, b8_char const& rhs){
        b8_char ret(0);
        ret.ch = lhs.ch ^ rhs.ch;
        return ret;
    }

    inline bool operator==(b8_char const& lhs, b8_char const& rhs){
        return lhs.ch == rhs.ch;
    }

    inline b8_ints operator*(b8_char const& lhs, int const& rhs){
        b8_ints ret;
        ret.b0 = lhs.b0 * rhs;
        ret.b1 = lhs.b1 * rhs;
        ret.b2 = lhs.b2 * rhs;
        ret.b3 = lhs.b3 * rhs;
        ret.b4 = lhs.b4 * rhs;
        ret.b5 = lhs.b5 * rhs;
        ret.b6 = lhs.b6 * rhs;
        ret.b7 = lhs.b7 * rhs;
        return ret;
    }

    inline b8_ints operator+(b8_char const& lhs, int const& rhs){
        b8_ints ret;
        ret.b0 = lhs.b0 + rhs;
        ret.b1 = lhs.b1 + rhs;
        ret.b2 = lhs.b2 + rhs;
        ret.b3 = lhs.b3 + rhs;
        ret.b4 = lhs.b4 + rhs;
        ret.b5 = lhs.b5 + rhs;
        ret.b6 = lhs.b6 + rhs;
        ret.b7 = lhs.b7 + rhs;
        return ret;
    }

    inline b8_ints operator+(b8_ints const& lhs, int const& rhs){
        b8_ints ret;
        ret.b0 = lhs.b0 + rhs;
        ret.b1 = lhs.b1 + rhs;
        ret.b2 = lhs.b2 + rhs;
        ret.b3 = lhs.b3 + rhs;
        ret.b4 = lhs.b4 + rhs;
        ret.b5 = lhs.b5 + rhs;
        ret.b6 = lhs.b6 + rhs;
        ret.b7 = lhs.b7 + rhs;
        return ret;
    }

    inline b8_ints operator-(b8_ints const& lhs, int const& rhs){
        return operator+(lhs,-rhs);
    }

    inline b8_ints operator+(b8_char const& lhs, b8_char const& rhs){
        b8_ints ret;
        ret.b0 = lhs.b0 + rhs.b0;
        ret.b1 = lhs.b1 + rhs.b1;
        ret.b2 = lhs.b2 + rhs.b2;
        ret.b3 = lhs.b3 + rhs.b3;
        ret.b4 = lhs.b4 + rhs.b4;
        ret.b5 = lhs.b5 + rhs.b5;
        ret.b6 = lhs.b6 + rhs.b6;
        ret.b7 = lhs.b7 + rhs.b7;
        return ret;
    }

    inline b8_ints operator+(b8_ints const& lhs, b8_char const& rhs){
        b8_ints ret;
        ret.b0 = lhs.b0 + rhs.b0;
        ret.b1 = lhs.b1 + rhs.b1;
        ret.b2 = lhs.b2 + rhs.b2;
        ret.b3 = lhs.b3 + rhs.b3;
        ret.b4 = lhs.b4 + rhs.b4;
        ret.b5 = lhs.b5 + rhs.b5;
        ret.b6 = lhs.b6 + rhs.b6;
        ret.b7 = lhs.b7 + rhs.b7;
        return ret;
    }

    inline b8_ints operator*(b8_ints const& lhs, int const& rhs){
        b8_ints ret;
        ret.b0 = lhs.b0 * rhs;
        ret.b1 = lhs.b1 * rhs;
        ret.b2 = lhs.b2 * rhs;
        ret.b3 = lhs.b3 * rhs;
        ret.b4 = lhs.b4 * rhs;
        ret.b5 = lhs.b5 * rhs;
        ret.b6 = lhs.b6 * rhs;
        ret.b7 = lhs.b7 * rhs;
        return ret;
    }

    inline b8_char operator>(b8_ints const& lhs, int const& rhs){
        b8_char ret;
        ret.b0 = lhs.b0 > rhs;
        ret.b1 = lhs.b1 > rhs;
        ret.b2 = lhs.b2 > rhs;
        ret.b3 = lhs.b3 > rhs;
        ret.b4 = lhs.b4 > rhs;
        ret.b5 = lhs.b5 > rhs;
        ret.b6 = lhs.b6 > rhs;
        ret.b7 = lhs.b7 > rhs;
        return ret;
    }


    inline std::ostream &operator<<(std::ostream &os, b8_char const &m) { 
        return os << m.b0 << m.b1 << m.b2 << m.b3 << m.b4 << m.b5 << m.b6 << m.b7;
    }
}
#endif //HDC_B8CHAR_HPP