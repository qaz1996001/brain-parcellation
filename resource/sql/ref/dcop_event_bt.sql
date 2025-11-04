CREATE DATABASE  IF NOT EXISTS `mes_omi` /*!40100 DEFAULT CHARACTER SET utf8 */;
USE `mes_omi`;
-- MySQL dump 10.13  Distrib 5.7.12, for Win32 (AMD64)
--
-- Host: 127.0.0.1    Database: mes_omi
-- ------------------------------------------------------
-- Server version	5.5.5-10.2.7-MariaDB

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `dcop_event_bt`
--

DROP TABLE IF EXISTS `dcop_event_bt`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `dcop_event_bt` (
  `VsPrimaryKey` varchar(36) NOT NULL,
  `tool_id` varchar(12) DEFAULT NULL COMMENT 'length extend to 12 for sintering "before/after" omi -- 2020/03/18 lkchena',
  `kind` int(11) DEFAULT NULL COMMENT '1:  RLY 2: Error 3: Event 6: SPC(bkm use) 9: system error -- 2019/10/29 lkchena\n\n//org:\n1: Information 2: Error 3: Wanming 6: SPC(bkm use) 9: system error -- 2019/10/29 lkchena',
  `code_name` varchar(12) DEFAULT NULL,
  `code_desc` varchar(64) DEFAULT NULL,
  `idxfield` int(11) DEFAULT NULL COMMENT 'bkm use: raw table(dcop_collect_bth) order of n01~64 / 0 for index',
  `event_cate` int(11) DEFAULT NULL COMMENT 'event category: 1: error start, 0: error end',
  `field_value` double DEFAULT NULL,
  `field_data` varchar(128) DEFAULT NULL COMMENT '128 ... 青志 powder_id''s length is 59 ...\r\nstring value for user definition, ex: powder change -- 2020/03/15 lkchena',
  `claim_time` varchar(19) DEFAULT NULL,
  `rec_time` varchar(19) DEFAULT NULL,
  `node_id` varchar(16) DEFAULT NULL,
  `node_time` timestamp(6) NULL DEFAULT current_timestamp(6) ON UPDATE current_timestamp(6),
  PRIMARY KEY (`VsPrimaryKey`),
  KEY `idx1` (`tool_id`,`field_data`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `dcop_event_bt`
--

LOCK TABLES `dcop_event_bt` WRITE;
/*!40000 ALTER TABLE `dcop_event_bt` DISABLE KEYS */;
INSERT INTO `dcop_event_bt` VALUES ('AF15T01_20191126181003264_1000_1_RE','AF15T01',0,'I1000','每小時補資料job',-1,1,1,'2019/11/26 18:10:03',NULL,'2019/11/26 18:10:03',NULL,'2020-02-16 01:18:35.284690'),('AF15T01_20191126181008420_1000_2_RE','AF15T01',0,'I1000','每小時補資料job',-1,1,9,'2019/11/26 18:10:08',NULL,'2019/11/26 18:10:08',NULL,'2020-02-16 01:18:35.284690'),('AF15T01_20191126191002464_1000_1_RE','AF15T01',0,'I1000','每小時補資料job',-1,1,1,'2019/11/26 19:10:02',NULL,'2019/11/26 19:10:02',NULL,'2020-02-16 01:18:35.284690'),('AF15T01_20191126191007791_1000_2_RE','AF15T01',0,'I1000','每小時補資料job',-1,1,9,'2019/11/26 19:10:07',NULL,'2019/11/26 19:10:07',NULL,'2020-02-16 01:18:35.284690'),('AF15T01_20191126193627958_1000_1_RE','AF15T01',0,'I1000','每小時補資料job',-1,1,1,'2019/11/26 19:36:27',NULL,'2019/11/26 19:36:27',NULL,'2020-02-16 01:18:35.284690'),('AF15T01_20191126193633597_1000_2_RE','AF15T01',0,'I1000','每小時補資料job',-1,1,9,'2019/11/26 19:36:33',NULL,'2019/11/26 19:36:33',NULL,'2020-02-16 01:18:35.284690'),('AF15T01_20191126201003229_1000_1_RE','AF15T01',0,'I1000','每小時補資料job',-1,1,1,'2019/11/26 20:10:03',NULL,'2019/11/26 20:10:03',NULL,'2020-02-16 01:18:35.284690'),('AF15T01_20191126201008509_1000_2_RE','AF15T01',0,'I1000','每小時補資料job',-1,1,9,'2019/11/26 20:10:08',NULL,'2019/11/26 20:10:08',NULL,'2020-02-16 01:18:35.284690'),('AF15T01_20191126211003271_1000_1_RE','AF15T01',0,'I1000','每小時補資料job',-1,1,1,'2019/11/26 21:10:03',NULL,'2019/11/26 21:10:03',NULL,'2020-02-16 01:18:35.284690'),('AF15T01_20191126211008549_1000_2_RE','AF15T01',0,'I1000','每小時補資料job',-1,1,9,'2019/11/26 21:10:08',NULL,'2019/11/26 21:10:08',NULL,'2020-02-16 01:18:35.284690'),('AF15T01_20191126211403_700_4_RE','AF15T01',3,'MR700','連續切',-1,1,2,'2019/11/26 21:14:03',NULL,'2019/11/26 22:10:08',NULL,'2020-02-16 01:18:35.284690');
/*!40000 ALTER TABLE `dcop_event_bt` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-05-12  8:16:33
