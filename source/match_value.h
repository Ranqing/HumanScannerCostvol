#ifndef MATCH_VALUE_H
#define MATCH_VALUE_H

class MatchValue {
public:
    MatchValue(const float d = 0.f, const float mcost = 0.f, const float prior = 0.f):m_d(d), m_mcost(mcost), m_prior(prior) {}
    ~MatchValue() {}

    float m_d, m_mcost, m_prior;
};

inline bool operator < (const MatchValue& oprand0, const MatchValue& oprand1) {
    return oprand0.m_mcost < oprand1.m_mcost;
}

inline bool operator == (const MatchValue& oprand0, const MatchValue& oprand1) {
    return oprand0.m_d == oprand1.m_d;
}

inline ostream & operator<<(ostream &os,const MatchValue &c) {
    os << c.m_d << ", " << c.m_mcost << ", " << c.m_prior << endl;

}

#endif // MATCH_VALUE_H
