#ifndef MATCH_H
#define MATCH_H

#include <iostream>
#include <vector>
using namespace std;

class Match
{
public:
    Match(const float x, const float y, const float d, const float mcost, const float prior);
    Match();
    ~Match();

    float get_mcost() const { return m_mcost; }
    float get_prior() const { return m_prior; }
    float get_x() const { return m_x; }
    float get_y() const { return m_y; }
    float get_d() const { return m_d; }

private:
    float m_x, m_y, m_d, m_mcost, m_prior;
};

inline Match::Match(const float x, const float y, const float d, const float mcost, const float prior):
    m_x(x), m_y(y), m_d(d), m_mcost(mcost), m_prior(prior) {

}

inline Match::Match() {

}

inline Match::~Match() {
}

inline bool operator < (const Match& oprand0, const Match& oprand1) {
    return oprand0.get_mcost() < oprand1.get_mcost();
}

inline bool operator == (const Match& oprand0, const Match& oprand1) {
    return oprand0.get_mcost() == oprand1.get_mcost();
}

inline ostream & operator<<(ostream &os,const Match &c) {
    os << c.get_x() << ", " << c.get_y() << ", " << c.get_d() << ", " << c.get_mcost()  << ", " << c.get_prior() << endl;

}

#endif // MATCH_H
