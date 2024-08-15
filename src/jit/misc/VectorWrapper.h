#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <stdexcept>
#include <vector>

/*
 * this class is a wrapper that treats multiple arrays a single one
 * random access is O(k), k being the number of arrays
*/
template <class T> class VectorWrapper {
private:
  size_t total_size = 0;
  std::vector<std::vector<T> *> vecList;

public:
  // empty default constructor
  VectorWrapper() {}

  template <typename... Vecs> VectorWrapper(Vecs &...vecL) {
    for (auto vec : {&vecL...}) {
      vecList.emplace_back(vec);
      total_size += vec->size();
    }
  }

  VectorWrapper(std::vector<T> &vec) { addVector(vec); }

  void addVector(std::vector<T> &vec) {
    vecList.emplace_back(&vec);
    total_size += vec.size();
  }

  T &at(size_t const pos) {

    if (vecList.size() == 0)
      throw std::out_of_range("Element not found. List is empty\n");

    size_t accumulated_size = 0;
    auto vec = vecList.begin();

    while (pos >= accumulated_size + (*vec)->size() && vec != vecList.end()) {
      accumulated_size += (*vec)->size();
      vec++;
    }

    if (vec == vecList.end())
      throw std::out_of_range("Position bigger than array size\n");

    return (*vec)->at(pos - accumulated_size);
  }

  T &operator[](size_t const pos) { return this->at(pos); }

  size_t size() const { return total_size; }

  // barebones iterator implementation. lacking a lot of funcionality
  struct Iterator {
    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type = ptrdiff_t;
    using value_type = T;
    using pointer = T *;
    using reference = T &;

    Iterator(VectorWrapper<T> &vw, size_t const pos)
        : myWrapper(vw), total_pos(pos) {
      calculateLimits();
    }

    Iterator(Iterator II, difference_type n)
        : Iterator(II.myWrapper, II.total_pos + n) {}

    Iterator &operator=(Iterator &other) {
      this->myWrapper = other.myWrapper;
      this->total_pos = other.total_pos;
      this->current_arr = 0;

      calculateLimits();
      return *this;
    }

    reference operator*() const { return myWrapper.at(total_pos); }
    pointer operator->() { return &myWrapper.at(total_pos); }

    Iterator &operator++() {
      if (++total_pos >= upperLimit and total_pos < myWrapper.size())
        incrementLimits(myWrapper.vecList[++current_arr]->size());
      return *this;
    }

    Iterator operator++(int) {
      return Iterator(myWrapper, this->total_pos + 1);
    }

    Iterator &operator--() {
      if (--total_pos < lowerLimit)
        decrementLimits(myWrapper.vecList[--current_arr]->size());
      return *this;
    }
    Iterator operator--(int) {
      return Iterator(myWrapper, this->total_pos - 1);
    }

    Iterator &operator+=(difference_type n) {
      total_pos += n;
      if (total_pos > total_pos - n)
        while (total_pos >= upperLimit and total_pos < myWrapper->size())
          IncrementLimits(myWrapper.vecList[++current_arr]->size());
      else
        while (total_pos <= lowerLimit and total_pos)
          decrementLimits(myWrapper.vecList[--current_arr]->size());

      return *this;
    }

    Iterator &operator-=(difference_type n) {
      total_pos -= n;
      if (total_pos > total_pos + n)
        while (total_pos >= upperLimit and total_pos < myWrapper->size())
          IncrementLimits(myWrapper.vecList[++current_arr]->size());
      else
        while (total_pos <= lowerLimit and total_pos)
          decrementLimits(myWrapper.vecList[--current_arr]->size());

      return *this;
    }

    friend bool operator==(const Iterator &a, const Iterator &b) {
      return a.total_pos == b.total_pos;
    }

    friend bool operator!=(const Iterator &a, const Iterator &b) {
      return a.total_pos != b.total_pos;
    }

    friend Iterator operator-(const Iterator &a, difference_type n) {
      return Iterator(a, -n);
    }

    friend difference_type operator-(const Iterator &a, const Iterator &b) {
      return a.total_pos - b.total_pos;
    }

    friend Iterator operator+(const Iterator &a, difference_type n) {
      return Iterator(a, n);
    }

    friend difference_type operator+(const Iterator &a, const Iterator &b) {
      return a.total_pos + b.total_pos;
    }

    friend bool operator<(const Iterator &a, const Iterator &b) {
      return a.total_pos < b.total_pos;
    }

    friend bool operator>(const Iterator &a, const Iterator &b) {
      return a.total_pos > b.total_pos;
    }

    friend bool operator<=(const Iterator &a, const Iterator &b) {
      return a.total_pos <= b.total_pos;
    }

    friend bool operator>=(const Iterator &a, const Iterator &b) {
      return a.total_pos >= b.total_pos;
    }

  private:
    void incrementLimits(int upperIncrement) {
      this->lowerLimit = this->upperLimit;
      this->upperLimit += upperIncrement;
    }
    void decrementLimits(int lowerDecrement) {
      this->upperLimit = this->lowerLimit;
      this->lowerLimit -= lowerDecrement;
    }

    void calculateLimits() {
      if (myWrapper.vecList.size() == 0)
        upperLimit = 0;
      else
        upperLimit = myWrapper.vecList[current_arr]->size();
      while (total_pos > upperLimit)
        incrementLimits(myWrapper.vecList[++current_arr]->size());
    }

    VectorWrapper<T> &myWrapper;

    int current_arr = 0;
    size_t upperLimit = 0;
    size_t lowerLimit = 0;
    size_t total_pos;
  };

  typedef Iterator iterator;

  iterator begin() { return iterator(*this, 0); }
  iterator end() { return iterator(*this, total_size); }
  iterator rbegin() { return iterator(*this, total_size - 1); }
  iterator rend() { return iterator(*this, 0); }
};
