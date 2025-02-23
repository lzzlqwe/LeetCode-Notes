# 哈希表

## [1.字母异位词分组](https://leetcode.cn/problems/group-anagrams/description/)
思路：创建一个HashMap，对每个字符串进行排序，排序后相同结果的归在同一个key下。  
代码：
```
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> mp = new HashMap<>();
        for(String str : strs){
            char[] carr = str.toCharArray();
            Arrays.sort(carr);
            String key = String.valueOf(carr);
            if(mp.containsKey(key))
                mp.get(key).add(str);
            else{
                List<String> list = new ArrayList<>();
                list.add(str);
                mp.put(key, list); 
            }
        }
        return new ArrayList<>(mp.values());
    }
}
```

## [2.最长连续序列](https://leetcode.cn/problems/longest-consecutive-sequence/description/)
思路：nums用一个hash表来存放，遍历hash表，如果存在比他小1的值，则考虑比他小的值即可，跳过；如果不存在比他小1的值，则说明当前值可能是最长连续序列的起点。  
代码：
```
class Solution {
    public int longestConsecutive(int[] nums) {
        //nums用一个hash表来存放
        Set<Integer> set = new HashSet<>();
        for(int x : nums) 
            set.add(x);
        int maxLength = 0;
        //遍历hash表
        for(int x : set){
            if(set.contains(x-1)) //如果存在比他小1的值，则考虑比他小的值即可。
                continue;
            int t = x; //不存在比他小1的值，说明当前值可能是最长连续序列的起点
            while(set.contains(t))
                t++;
            maxLength = Math.max(maxLength, t - x); //更新答案
        }
        return maxLength;
    }
}
```

# 双指针

## [1.移动零](https://leetcode.cn/problems/move-zeroes/description/)
思路：使用双指针，左指针始终指向第一个0， 右指针指向left后的第一个非零元素（其中left到right-1都为0），进行元素交换后，双指针继续向后移动。  
代码：
```
class Solution {
    public void moveZeroes(int[] nums) {
        int n = nums.length;
        int left = 0; //始终指向第一个0
        while(left < n && nums[left] != 0)
            left++;
        int right = left; //指向left后的非零元素
        while(right < n && nums[right] == 0)
            right++;
        while(right < n){
            int temp = nums[left];
            nums[left] = nums[right];
            nums[right] = temp;
            while(right < n && nums[right] == 0)
                right++;
            while(left < n && nums[left] != 0)
                left++;
        }
    }
}
```

## [2.盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/description/)
思路：定义两个指针left和right，left从最左边开始，right从最右边开始，每次移动最矮的那条边（因为移动最矮边才可能在中间找到最优解），并记录储存的最大水量。  
代码：
```
class Solution {
    public int maxArea(int[] height) {
        int ans = 0;
        int left = 0;
        int right = height.length - 1;
        while(left < right){
            int s = (right - left) * Math.min(height[left], height[right]);
            ans = Math.max(ans, s);
            if(height[left] < height[right])
                left++;
            else
                right--;
        }
        return ans;
    }
}
```

## [3.三数之和](https://leetcode.cn/problems/3sum/description/)
思路：先排序，然后定义三个指针i，left和right，外层循环指针i从0遍历到倒数第三个位置（留两个位置给相向指针left和right），指针left从i+1开始，指针right从最右边开始。（如果题目要求结果不能重复，则三个指针都只指向第一个重复元素，绕过后续重复元素）  
代码：
```
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums); //排序
        List<List<Integer>> ans = new ArrayList<>();
        int n = nums.length;
        int i = 0;
        while(i < n - 2){
            int left = i + 1;
            int right = n - 1;
            while(left < right){
                if(nums[i] + nums[left] + nums[right] == 0){
                    ans.add(new ArrayList(Arrays.asList(nums[i], nums[left], nums[right])));
                    left++;
                    while(left < right && nums[left-1] == nums[left])//指向第一个重复的元素
                        left++;
                    right--;
                    while(left < right && nums[right] == nums[right+1])//指向第一个重复的元素
                        right--;
                }
                else if(nums[i] + nums[left] + nums[right] < 0)
                    left++;
                else
                    right--;
            }
            i++;
            while(i < n - 2 && nums[i-1] == nums[i]) //指向第一个重复的元素,绕过后续重复元素
                i++;
        }
        return ans;
    }
}
```

## [4.接雨水](https://leetcode.cn/problems/trapping-rain-water/description/)
思路：定义两个数组，前缀数组记录从左边到下标i的最大高度，后缀数组记录从右边到下标i的最大高度，有了这两个数组就可以计算各个下标i的盛水单位量。  
代码：
```
class Solution {
    public int trap(int[] height) {
        int n = height.length;
        int[] pre = new int[n];
        int[] post = new int[n];
        //求前缀数组和后缀数组
        int max = 0;
        for(int i = 0; i < n; i++){
            max = Math.max(max, height[i]);
            pre[i] = max;
        }
        max = 0;
        for(int i = n-1; i >= 0; i--){
            max = Math.max(max, height[i]);
            post[i] = max;
        }
        //根据前缀数组和后缀数组求答案
        int ans = 0;
        for(int i = 0; i < n; i++){
            ans += Math.min(pre[i], post[i]) - height[i];
        }
        return ans;
    }
}
```

# 滑动窗口

### 模板代码
```
int start = 0;
int end = 0;
while(end < n){ //外层循环扩展右边界，内层循环扩展左边界
    xxx;  //当前考虑的元素
    while(start <= end && 条件){
        xxx
        start++; //扩展左边界
    }
    end++; //扩展右边界
}
```

### 滑动窗口使用前提：  
1. 连续子数组。
2. 有单调性。比如元素均为正数。

## [1.无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/description/)
思路：套用滑动窗口模板  
代码：
```
class Solution {
    public int lengthOfLongestSubstring(String s) {
        Set<Character> set = new HashSet<>();
        char[] strArr = s.toCharArray();
        int n = strArr.length;
        int left = 0;
        int right = 0;
        int maxLength = 0;
        while(right < n){
            char c = strArr[right];
            while(left <= right && set.contains(c)){
                set.remove(strArr[left]);
                left++;
            }
            set.add(c);
            maxLength = Math.max(maxLength, right - left + 1);
            right++;
        }
        return maxLength;
    }
}
```

## [2.找到字符串中所有字母异位词](https://leetcode.cn/problems/find-all-anagrams-in-a-string/description/)
思路：首先统计字符串p的各个字母出现的次数，然后套用滑动窗口模板，统计滑动窗口中各字符的次数，如果滑动窗口中当前字符的次数大于字符串p的字符次数，则移动滑动窗口左边。当滑动窗口的长度等于字符串p的长度时，即找到一个答案。之后继续查找。  
代码：
```
class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        List<Integer> ans = new ArrayList<>();
        //统计字符串p的各个字母出现的次数
        char[] p_mp = new char[26];
        char[] pArr = p.toCharArray();
        for(char c : pArr){
            p_mp[c-'a']++;
        }
        //找到异位词
        char[] s_mp = new char[26];
        int left = 0;
        int right = 0;
        char[] sArr = s.toCharArray();
        int n = sArr.length;
        while(right < n){
            char c = sArr[right]; //考虑当前字符
            // s_mp.merge(c, 1, Integer::sum); 
            s_mp[c-'a']++; //添加到hashmap
            //如果滑动窗口中当前字符的次数大于字符串p的字符次数，则移动滑动窗口左边
            while(left <= right && s_mp[c-'a'] > p_mp[c-'a']){ 
                s_mp[sArr[left]-'a']--;
                left++;
            }
            if(right - left + 1 == p.length()) //找到答案
                ans.add(left);
            right++;
        }
        return ans;
    }
}
```

# 子串

## [1.和为 K 的子数组](https://leetcode.cn/problems/subarray-sum-equals-k/description/)
思路：前缀和pre[i]表示[0...i]的和。和为k的子数组nums[j, i]满足pre[i] - pre[j-1] = k，即pre[i] - k = pre[j-1]。因此转换思路，在for循环中，查找以i结尾的和为k的子数组个数时，只需计算前面有多少个和为pre[i] - k的子数组。  
代码：
```
class Solution {
    public int subarraySum(int[] nums, int k) {
        //前缀和pre[i]表示[0...i]的和
        //和为k的子数组nums[j, i]满足pre[i] - pre[j-1] = k，即pre[i] - k = pre[j-1]
        //因此for循环，查找以i结尾的和为k的子数组个数，只需计算前面有多少个和为pre[i] - k的子数组
        int ans = 0; //记录答案
        int pre = 0; //记录前缀和
        Map<Integer, Integer> mp = new HashMap<>(); //记录前缀和的出现个数
        mp.put(0, 1); //初始化，pre[-1] = 0;
        for(int i = 0; i < nums.length; i++){
            pre += nums[i]; //更新前缀和
            if(mp.containsKey(pre - k))
                ans += mp.get(pre - k);
            mp.merge(pre, 1, Integer::sum);
        }
        return ans;
    }
}
```

## [2.滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/description/)
思路：从左到右，维护一个单调队列，队首元素就是当前滑动窗口的最大值，详见代码注释。  
代码：
```
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        List<Integer> dequeue = new LinkedList<>(); //单调队列
        for(int i = 0; i < k-1; i++){ //初始维护单调队列
            while(!dequeue.isEmpty() && nums[dequeue.getLast()] <= nums[i])
                dequeue.removeLast();
            dequeue.addLast(i); //存放下标而不是值
        }
        //移动滑动窗口得到答案
        int n = nums.length;
        int[] ans = new int[n-k+1];
        for(int i = k-1; i < n; i++){
            //维护单调队列
            while(!dequeue.isEmpty() && nums[dequeue.getLast()] <= nums[i])
                dequeue.removeLast();
            dequeue.addLast(i); //存放下标而不是值
            //检查队首是否已经超出滑动窗口了
            while(dequeue.getFirst() < i - k + 1)
                dequeue.removeFirst();
            //队首元素即为当前滑动窗口最大值
            ans[i-k+1] = nums[dequeue.getFirst()];
        }
        return ans;
    }
}
```


## [3.最小覆盖子串](https://leetcode.cn/problems/minimum-window-substring/description/)
思路：属于滑动窗口专题，和滑动窗口专题的第二题很相似。虽然存在双重循环，但时间复杂度为并不是O(n^2)，而是O(C·n)，其中C是一个常数。因为这里内层循环遍历的是52个大小写字母，是固定的常数。
代码：
```
class Solution {
    public String minWindow(String s, String t) {
        char[] t_mp = new char[128]; //t的哈希表
        char[] tArr = t.toCharArray();
        for(char c : tArr){
            t_mp[c - 'A']++;
        }
        char[] s_mp = new char[128]; //s的哈希表
        char[] sArr = s.toCharArray();
        int left = 0;
        int right = 0;
        int n = sArr.length;
        int[] ans = new int[]{0, 100001}; //记录最小覆盖子串的起始下标和长度
        while(right < n){
            char c = sArr[right]; //考虑当前字符
            s_mp[c - 'A']++; //加入hash表
            while(left <= right && isCover(t_mp, s_mp)){
                if(right - left + 1 < ans[1]){ //更新并记录答案
                    ans[0] = left;
                    ans[1] = right - left + 1;
                }
                s_mp[sArr[left] - 'A']--;
                left++;
            }
            right++;
        }
        if(ans[1] > 100000) return "";
        else return s.substring(ans[0], ans[0]+ans[1]);
    }

    //遍历52个字母，检查当前滑动窗口是否覆盖了字符串t
    boolean isCover(char[] t_mp, char[] s_mp){
        for(int i = 0; i < 26; i++) //大写字母
            if(t_mp[i] > s_mp[i])
                return false;
        for(int i = 0; i < 26; i++) //小写字母
            if(t_mp[i+'a'-'A'] > s_mp[i+'a'-'A'])
                return false;
        return true;
    }
}
```
